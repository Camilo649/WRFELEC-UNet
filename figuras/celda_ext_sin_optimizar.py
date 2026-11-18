# -*- coding: utf-8 -*-
# =====================================================================
# CELDA — estimaciones de la Ext Clip U-Net FACL SIN OPTIMIZAR
# (para justificar en el Cap. 6 por qué no se optimizó la Comb V2.1)
#
# Dónde pegarla: en la misma notebook donde corriste
# celda_resultados_validacion.py (Entrenamiento_MemoFatShortUnetFACL),
# en cualquier lugar después de esa celda. Usa `UNet_WRF`,
# `TormentaDataset` y `device`, que ya están en la notebook, y lee
# resultados_validacion.npz (el que generó esa celda) para usar las
# mismas 2.271 muestras, en el mismo orden.
#
# Qué hace: corre sobre la validación el checkpoint
#   ../modelos/20260929_0332_Fat Short Unet FACL.pth
# (la Ext Clip U-Net FACL antes de la optimización: 3 niveles, 64
# filtros base, kernels 5, 3 y 3, es decir, la misma arquitectura que
# la Ext Clip U-Net Comb V2.1), con los pesos de
#   - su mejor época (best_model_state_dict, la 5 según el
#     checkpoint), y
#   - su última época (model_state_dict, la 25),
# y guarda las dos estimaciones (desnormalizadas, limitadas a [0, 1]
# antes de desnormalizar y redondeadas, igual que las otras) en
# pred_ext_sin_optimizar.npz.
#
# Se exportan las dos épocas porque la Ext Clip U-Net FACL optimizada
# se evalúa con su última época (su pérdida de validación, L_val, no
# servía para elegir la mejor), mientras que la Comb V2.1 se evalúa
# con su mejor época. Las métricas las calculo yo con el mismo código
# de la Tabla 6.1.
#
# Tarda unos minutos con GPU. Al terminar, mandame
# pred_ext_sin_optimizar.npz y lo que imprime la celda.
# =====================================================================
import os

import numpy as np
import torch
from tqdm.auto import tqdm

RUTA_CKPT = "../modelos/20260929_0332_Fat Short Unet FACL.pth"
ARQ = dict(n_niveles=3, n_filtros_base=64, kernel_sizes=[5, 3, 3],
           dropout_ps=None)           # el dropout no influye al evaluar
DIR_EXT = "../dataset"                # conjunto extendido, 3 canales
RUTA_REF = "resultados_validacion.npz"
RUTA_SALIDA = "pred_ext_sin_optimizar.npz"
LOTE = 16
TAM = 297

# --- 1) Muestras: las mismas y en el mismo orden que la Tabla 6.1 ----
ref = np.load(RUTA_REF)
nombres = [str(n) for n in ref["archivos"]]
real = ref["y_real"]
ds = TormentaDataset(split_dir=f"{DIR_EXT}/val",
                     stats_path=f"{DIR_EXT}/stats.npz")
idx = {f.name: i for i, f in enumerate(ds.files)}
faltan = [n for n in nombres if n not in idx]
assert not faltan, f"faltan {len(faltan)} muestras en {DIR_EXT}/val"
print(f"Muestras: {len(nombres)} (esperado: 2271)")


def _a_entero(t):
    """Desnormaliza un mapa y lo redondea, como en las métricas."""
    v = np.round(ds.desnormalizar_y(t).numpy())
    assert v.min() >= 0 and v.max() <= 255, (v.min(), v.max())
    return v.astype(np.uint8)


# control: el valor simulado tiene que coincidir con el de la Tabla 6.1
for k in (0, len(nombres) // 2, len(nombres) - 1):
    y = _a_entero(ds[idx[nombres[k]]][1].squeeze())
    assert (y == real[k]).all(), f"{nombres[k]}: valor simulado distinto"

# --- 2) Modelo ---------------------------------------------------------
ck = torch.load(RUTA_CKPT, map_location=device, weights_only=False)
print(f"Checkpoint: mejor época {ck.get('mejor_epoca')} de "
      f"{ck.get('epochs_entrenadas')} | mejor val_loss "
      f"{ck.get('mejor_val_loss')}")


def _modelo(clave):
    sd = ck[clave]
    m = UNet_WRF(in_channels=next(iter(sd.values())).shape[1],
                 **ARQ).to(device)
    m.load_state_dict(sd)
    n = sum(p.numel() for p in m.parameters())
    print(f"  {clave}: {n:,} parámetros (la Comb V2.1 tiene 8,113,665)")
    return m.eval()


def _predecir(m, desc):
    out = np.zeros((len(nombres), TAM, TAM), dtype=np.uint8)
    with torch.no_grad():
        for a in tqdm(range(0, len(nombres), LOTE), desc=desc):
            lote = nombres[a:a + LOTE]
            X = torch.stack([ds[idx[n]][0] for n in lote]).to(device)
            p = m(X).cpu().squeeze(1).clamp(0, 1)
            for k in range(len(lote)):
                out[a + k] = _a_entero(p[k])
    return out


salida = dict(archivos=np.array(nombres))
salida["pred_sin_opt_mejor"] = _predecir(_modelo("best_model_state_dict"),
                                         "Sin optimizar, mejor época")
salida["pred_sin_opt_ultima"] = _predecir(_modelo("model_state_dict"),
                                          "Sin optimizar, última época")


# --- 3) Control rápido --------------------------------------------------
def _csi_medio(r, p):
    vals = []
    for a, b in zip(r, p):
        act = (a > 0) | (b > 0)
        if act.any():
            vals.append(((a > 0) & (b > 0)).sum() / act.sum())
    return float(np.mean(vals))


print("\nCSI promedio sobre los pasos de tiempo:")
print(f"  Comb V2.1 (referencia)       : "
      f"{_csi_medio(real, ref['pred_comb_mejor']):.3f} (esperado: 0.577)")
for clave in ("pred_sin_opt_mejor", "pred_sin_opt_ultima"):
    print(f"  {clave:29s}: {_csi_medio(real, salida[clave]):.3f}")

np.savez_compressed(RUTA_SALIDA, **salida)
print(f"\nGuardado {RUTA_SALIDA} "
      f"({os.path.getsize(RUTA_SALIDA) / 1e6:.1f} MB). "
      "Mandame ese archivo y esta salida.")
