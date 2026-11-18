# -*- coding: utf-8 -*-
# =====================================================================
# CELDA 1 DE RESULTADOS (Capítulo 6) — exporta las estimaciones de
# todo el conjunto de validación.
#
# Dónde pegarla: AL FINAL de Entrenamiento_Temp_Ext_Clip_U-Net_FACL.ipynb,
# después de la celda "CHECKPOINT DE LA ETAPA B OFICIAL — ensayo 100,
# mejor época (best.pt)", que deja armados `modelo` (el modelo final,
# en .eval()) y `val_dataset` (../dataset_temporal/val, 9 canales).
# Usa además `UNet_WRF`, `TormentaDataset` y `device`, que ya están en
# la notebook.
#
# Qué hace: corre sobre las 2.271 muestras de validación
#   - el modelo final (Temp Ext Clip U-Net FACL, ensayo 100, época 47),
#   - la Ext Clip U-Net FACL con los pesos de su última época (la
#     140) del ensayo 39 reproducido con
#     `celda_curvas_fal_fcl_trial39.py` (clave `modelo` del
#     checkpoint ../modelos/etapa_B_Ext_Clip_U-Net_FACL/
#     B_trial0039_repro_curvas.pth),
#   - la Ext U-Net Comb V2.1 con los pesos de su mejor época
#     (para rehacer la figura de comparación),
# y guarda el valor simulado y las tres estimaciones
# (desnormalizadas y redondeadas, como en las métricas) en
# resultados_validacion.npz.
#
# OJO (versión 3): la Ext Clip U-Net FACL se carga del checkpoint de
# la reproducción (ya no del ensayo original) y no se evalúa su
# mejor época, porque la reproducción no guardó esos pesos.
#
# Tarda unos minutos con GPU. Al terminar, mandame
# resultados_validacion.npz y lo que imprime la celda.
# =====================================================================
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

# --- 1) Rutas ---------------------------------------------------------
RUTA_EXT = ("../modelos/etapa_B_Ext_Clip_U-Net_FACL/"
            "B_trial0039_repro_curvas.pth")
# Arquitectura del ensayo 106 (el dropout no influye al evaluar)
ARQ_EXT = dict(n_niveles=3, n_filtros_base=64,
               kernel_sizes=[5, 5, 3],
               dropout_ps=[0.024461173266027245, 0.07965792664013316,
                           0.27383264907951727])
RUTA_COMB = "../modelos/20260924_0326_Ext U-Net Comb V2.1.pth"
ARQ_COMB = dict(n_niveles=3, n_filtros_base=64,
                kernel_sizes=[5, 3, 3], dropout_ps=None)
DIR_EXT = "../dataset"           # conjunto extendido, 3 canales
LOTE = 16
TAM = 297
RUTA_SALIDA = "resultados_validacion.npz"

# --- 2) Chequeo del modelo final --------------------------------------
n_par = sum(p.numel() for p in modelo.parameters())
print(f"Modelo final: {n_par:,} parámetros (esperado: 5,410,241)")
assert n_par == 5_410_241, "`modelo` no es el ensayo 100"


# --- 3) Modelos del conjunto extendido (3 canales) --------------------
def _cargar(ruta, clave, arq, sigmoide=True):
    """Arma el modelo con los pesos guardados en `clave`:
    'best_model_state_dict' / 'model_state_dict' (checkpoints
    anteriores) o 'modelo' (checkpoint de la reproducción)."""
    ck = torch.load(ruta, map_location=device, weights_only=False)
    assert clave in ck, f"{ruta} no tiene {clave}: {list(ck.keys())}"
    sd = ck[clave]
    cin = next(iter(sd.values())).shape[1]
    m = UNet_WRF(in_channels=cin, **arq).to(device)
    m.load_state_dict(sd)
    if not sigmoide:            # la Comb V2.1 no tiene sigmoide
        m.sigmoid = nn.Identity()
    n = sum(p.numel() for p in m.parameters())
    print(f"  {clave}: {cin} canales, {n:,} parámetros | mejor época "
          f"{ck.get('mejor_epoca')} de {ck.get('epochs_entrenadas')}")
    return m.eval()


print("Ext Clip U-Net FACL:", RUTA_EXT)
ext_ultima = _cargar(RUTA_EXT, "modelo", ARQ_EXT)
print("Ext U-Net Comb V2.1:", RUTA_COMB)
comb_mejor = _cargar(RUTA_COMB, "best_model_state_dict", ARQ_COMB,
                     sigmoide=False)

# --- 4) Muestras, alineadas por nombre de archivo ---------------------
ds_t = getattr(val_dataset, "dataset", val_dataset)     # 9 canales
ds_e = TormentaDataset(split_dir=f"{DIR_EXT}/val",
                       stats_path=f"{DIR_EXT}/stats.npz")
idx_t = {f.name: i for i, f in enumerate(ds_t.files)}
idx_e = {f.name: i for i, f in enumerate(ds_e.files)}
nombres = sorted(set(idx_t) & set(idx_e))
print(f"Muestras: {len(idx_t)} (temporal), {len(idx_e)} (extendido), "
      f"{len(nombres)} en común (esperado: 2271)")


def _a_entero(ds, t):
    """Desnormaliza un mapa y lo redondea, como en las métricas."""
    v = np.round(ds.desnormalizar_y(t).numpy())
    assert v.min() >= 0 and v.max() <= 255, (v.min(), v.max())
    return v.astype(np.uint8)


def _predecir(m, ds, idx, desc):
    out = np.zeros((len(nombres), TAM, TAM), dtype=np.uint8)
    m.eval()
    with torch.no_grad():
        for a in tqdm(range(0, len(nombres), LOTE), desc=desc):
            lote = nombres[a:a + LOTE]
            X = torch.stack([ds[idx[n]][0] for n in lote]).to(device)
            p = m(X).cpu().squeeze(1).clamp(0, 1)
            for k in range(len(lote)):
                out[a + k] = _a_entero(ds, p[k])
    return out


# --- 5) Valor simulado y estimaciones ---------------------------------
real = np.zeros((len(nombres), TAM, TAM), dtype=np.uint8)
for k, n in enumerate(tqdm(nombres, desc="WRF-ELEC")):
    real[k] = _a_entero(ds_t, ds_t[idx_t[n]][1].squeeze())
# control: el valor simulado debería ser el mismo en los dos
# conjuntos (no frena la celda; solo avisa si difieren)
for k in (0, len(nombres) // 2, len(nombres) - 1):
    otro = _a_entero(ds_e, ds_e[idx_e[nombres[k]]][1].squeeze())
    dif = int((real[k] != otro).sum())
    if dif:
        print(f"!! {nombres[k]}: el valor simulado difiere en {dif} "
              "columnas entre ../dataset y ../dataset_temporal")

salida = dict(archivos=np.array(nombres), y_real=real)
salida["pred_temp"] = _predecir(modelo, ds_t, idx_t,
                                "Temp Ext Clip U-Net FACL")
salida["pred_ext_ultima"] = _predecir(ext_ultima, ds_e, idx_e,
                                      "Ext Clip U-Net FACL (última época)")
salida["pred_comb_mejor"] = _predecir(comb_mejor, ds_e, idx_e,
                                      "Ext Comb V2.1 (mejor)")


# --- 6) Control rápido contra la Tabla 6.1 ----------------------------
def _csi_medio(r, p):
    vals = []
    for a, b in zip(r, p):
        act = (a > 0) | (b > 0)
        if act.any():
            vals.append(((a > 0) & (b > 0)).sum() / act.sum())
    return float(np.mean(vals))


print("\nControl (CSI promedio sobre los pasos de tiempo):")
for clave, nombre, esperado in (
        ("pred_temp", "Temp Ext Clip U-Net FACL", "0.659"),
        ("pred_ext_ultima", "Ext Clip U-Net FACL, última época",
         "≈0.659 (pesos viejos)"),
        ("pred_comb_mejor", "Ext U-Net Comb V2.1, mejor", "-")):
    print(f"  {nombre:36s}: {_csi_medio(real, salida[clave]):.3f}  "
          f"(esperado: {esperado})")

np.savez_compressed(RUTA_SALIDA, **salida)
print(f"\nGuardado {RUTA_SALIDA} "
      f"({os.path.getsize(RUTA_SALIDA) / 1e6:.1f} MB). "
      "Mandame ese archivo y esta salida.")
