# -*- coding: utf-8 -*-
# =====================================================================
# CELDA PARA LA NOTEBOOK (pegar al final de
# Entrenamiento_MemoFatShortUnetFACL.ipynb, o de cualquier notebook
# que ya tenga importados UNet_WRF, TormentaDataset, device, np,
# torch y tqdm).
#
# Qué hace: corre varios checkpoints sobre el MISMO instante de
# validación y guarda las predicciones (ya desnormalizadas y
# redondeadas, igual que en graficar_reconstruccion) en un .npz.
# Después, figuras/generar_figuras_modelos.py arma la figura con
# todos los modelos.
#
# Pasos:
#   1. Completar las rutas de MODELOS (solo hay que revisar la del
#      Optimized: el resto está tomado de tus notebooks y del ls).
#   2. Dejar INSTANTE = None para que la celda proponga los mejores
#      instantes, mirar la lista que imprime y volver a correr con
#      INSTANTE = '<nombre del archivo>' (o un índice).
#   3. Copiar comparacion_instante.npz a figuras/datos/ y correr
#      python3 figuras/generar_figuras_modelos.py
#
# Todos los modelos de esta celda usan el conjunto ampliado y
# saturado (../dataset, 3 canales; ../dataset_temporal, 9 canales).
# Los modelos anteriores (6 variables, otros conjuntos) no se
# pueden correr sobre estas muestras.
# =====================================================================
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

# --- 1) Modelos -------------------------------------------------------
ARQ_FAT = dict(n_niveles=3, n_filtros_base=64,
               kernel_sizes=[5, 3, 3], dropout_ps=None)
# Arquitectura del ensayo 106 (el dropout no influye al evaluar)
ARQ_OPT = dict(n_niveles=3, n_filtros_base=64,
               kernel_sizes=[5, 5, 3], dropout_ps=[0.03, 0.08, 0.26])

MODELOS = [
    # nombre en la figura, checkpoint, carpeta del dataset,
    # ¿lleva sigmoide a la salida?, arquitectura si el checkpoint
    # no la trae (None: la lee de ckpt['arquitectura'])
    dict(nombre="Fat Short U-Net Comb V2.2",
         ruta="../modelos/20260924_0326_Fat Short Unet Comb V2.2.pth",
         dir="../dataset", sigmoide=False, arq=ARQ_FAT),
    dict(nombre="Fat Short U-Net FACL",
         ruta="../modelos/20260929_0332_Fat Short Unet FACL.pth",
         dir="../dataset", sigmoide=True, arq=ARQ_FAT),
    dict(nombre="Fat Short U-Net FACL (Optimized)",
         ruta="../modelos/etapa_B_Fat_Short_Unet_FACL/"
              "B_trial0039_Fat Short Unet FACL.pth",
         dir="../dataset", sigmoide=True, arq=ARQ_OPT),
    dict(nombre="Memo Fat Short U-Net FACL",
         ruta="../modelos/etapa_temporal_FACL/checkpoints/"
              "temporal_1545855_best.pth",
         dir="../dataset_temporal", sigmoide=True, arq=None),
    # Cuando exista el Memo optimizado, agregarlo acá:
    # dict(nombre="Memo Fat Short U-Net FACL (Optimized)", ...),
]

INSTANTE = None          # None, un nombre de archivo o un índice
MIN_PIXELES = 300        # para proponer: píxeles con actividad real
MIN_MAXIMO = 60          # para proponer: máximo real de LIGHTDENS
MARGEN = 30              # para proponer: se cuentan solo los píxeles
                         # a más de MARGEN del borde del dominio
RUTA_SALIDA = "comparacion_instante.npz"


def _csi(real, pred):
    m = (real > 0) | (pred > 0)
    if m.sum() == 0:
        return np.nan
    o, p = real[m] > 0, pred[m] > 0
    return float((o & p).sum() / m.sum())


def _cargar(cfg):
    ck = torch.load(cfg["ruta"], map_location=device,
                    weights_only=False)      # checkpoint propio
    sd = ck["model_state_dict"] if (isinstance(ck, dict) and
                                    "model_state_dict" in ck) else ck
    arq = cfg["arq"] if cfg["arq"] is not None else ck["arquitectura"]
    cin = ck["in_channels"] if (isinstance(ck, dict) and
                                "in_channels" in ck) \
        else next(iter(sd.values())).shape[1]
    m = UNet_WRF(in_channels=cin, **arq).to(device)
    m.load_state_dict(sd)
    if not cfg["sigmoide"]:
        m.sigmoid = nn.Identity()
    n_par = sum(p.numel() for p in m.parameters())
    print(f"{cfg['nombre']}: {cin} canales, {n_par:,} parámetros")
    return m.eval()


def _pred(m, ds, i):
    X, _ = ds[i]
    with torch.no_grad():
        p = m(X.unsqueeze(0).to(device)).cpu().squeeze()
    p = p.clamp(0, 1)   # rango normalizado; los modelos sin sigmoide
                        # pueden pasarse y exp() lo amplifica
    return np.round(ds.desnormalizar_y(p).numpy())


# --- 2) Datasets y modelos --------------------------------------------
import glob
dss, mods, usados = {}, [], []
for cfg in MODELOS:
    if "COMPLETAR" in cfg["ruta"]:
        carpeta = cfg["ruta"].rsplit("/", 1)[0]
        print(f"!! {cfg['nombre']}: falta la ruta del checkpoint. "
              f"Archivos .pth encontrados en {carpeta}:")
        for f in sorted(glob.glob(f"{carpeta}/**/*.pth",
                                  recursive=True)):
            print("     ", f)
        continue
    try:
        m = _cargar(cfg)
    except Exception as e:      # no frenar toda la celda por uno
        print(f"!! No se pudo cargar '{cfg['nombre']}': {e}")
        continue
    if cfg["dir"] not in dss:
        dss[cfg["dir"]] = TormentaDataset(
            split_dir=f"{cfg['dir']}/val",
            stats_path=f"{cfg['dir']}/stats.npz")
    mods.append(m)
    usados.append(cfg)
MODELOS = usados          # los omitidos quedan "pendiente" en la figura

# Las muestras se alinean por nombre de archivo (el dataset temporal
# tiene las mismas muestras, pero conviene no depender del orden).
indice = {d: {f.name: i for i, f in enumerate(ds.files)}
          for d, ds in dss.items()}
comunes = set.intersection(*[set(v) for v in indice.values()])
print(f"Muestras de validación comunes: {len(comunes)}")

ds0 = dss[MODELOS[0]["dir"]]
nombres_archivo = sorted(comunes)

# --- 3) Elegir el instante ---------------------------------------------
if INSTANTE is None:
    filas = []
    for a in tqdm(nombres_archivo, desc="Buscando un buen instante"):
        _, y = ds0[indice[MODELOS[0]["dir"]][a]]
        real = np.round(ds0.desnormalizar_y(y.squeeze()).numpy())
        centro = real[MARGEN:-MARGEN, MARGEN:-MARGEN]
        if (centro > 0).sum() < MIN_PIXELES or real.max() < MIN_MAXIMO:
            continue
        cs = [_csi(real, _pred(m, dss[c["dir"]],
                               indice[c["dir"]][a]))
              for m, c in zip(mods, MODELOS)]
        filas.append((np.nanmean(cs), a, cs))
    filas.sort(reverse=True)
    print("\nMejores instantes (CSI medio de los modelos):")
    for sc, a, cs in filas[:10]:
        print(f"  {sc:.3f}  {a}   " +
              "  ".join(f"{c:.2f}" for c in cs))
    print("\nElegí uno y volvé a correr con INSTANTE = '<archivo>'.")
else:
    a = (nombres_archivo[INSTANTE] if isinstance(INSTANTE, int)
         else INSTANTE)
    _, y = ds0[indice[MODELOS[0]["dir"]][a]]
    real = np.round(ds0.desnormalizar_y(y.squeeze()).numpy())
    preds, csis = [], []
    for m, c in zip(mods, MODELOS):
        p = _pred(m, dss[c["dir"]], indice[c["dir"]][a])
        preds.append(p.astype("float32"))
        csis.append(_csi(real, p))
        print(f"{c['nombre']}: CSI = {csis[-1]:.3f}, "
              f"máx = {p.max():.0f} (real {real.max():.0f})")
    np.savez(RUTA_SALIDA,
             nombres=np.array([c["nombre"] for c in MODELOS]),
             y_real=real.astype("float32"),
             preds=np.stack(preds), csi=np.array(csis),
             archivo=a)
    print(f"Guardado en {RUTA_SALIDA} ({a})")
