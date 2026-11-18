# -*- coding: utf-8 -*-
# =====================================================================
# CELDA PARA Entrenamiento_Ext_Clip_U-Net_FACL.ipynb
#
# Qué hace: estima las 2.271 muestras de validación con la Ext Clip
# U-Net FACL con los pesos de la última época (la 140) del ensayo 39
# reproducido (B_trial0039_repro_curvas.pth). Las estimaciones de los
# otros modelos (Temp y Comb V2.1) ya las tengo y no cambian.
#
# Antes de correrla, ejecutá las celdas de: Dependencias, UNet_WRF,
# DATASETS (define TormentaDataset). No hace falta la celda de las
# curvas: la arquitectura está escrita en esta celda.
#
# Tarda unos minutos con GPU.
# Salida: ../estudios/pred_ext_ultima_repro.npz  <- mandame este y lo
#         que imprime la celda.
# =====================================================================
import os

import numpy as np
import torch
from tqdm.auto import tqdm

CKPT = ('../modelos/etapa_B_Ext_Clip_U-Net_FACL/'
        'B_trial0039_repro_curvas.pth')
DIR_EXT = '../dataset'            # conjunto extendido, 3 canales
LOTE, TAM = 16, 297
SALIDA = '../estudios/pred_ext_ultima_repro.npz'

# Arquitectura (ensayo 106 de la Fase 1) e hiperparámetros del ensayo
# 39 de la Fase 2, tal como quedaron en curvas_fal_fcl_trial0039.json
ARQ = dict(n_niveles=3, n_filtros_base=64, kernel_sizes=[5, 5, 3],
           dropout_ps=[0.024461173266027245, 0.07965792664013316,
                       0.27383264907951727])
alpha, epocas = 0.18908467719664426, 140

ck = torch.load(CKPT, map_location=device, weights_only=False)
assert ck['epoca'] == 140, 'el entrenamiento no terminó'
m = UNet_WRF(in_channels=3, **ARQ).to(device)
m.load_state_dict(ck['modelo'])
m.eval()
print(f"Ext Clip U-Net FACL: {sum(p.numel() for p in m.parameters()):,}"
      " parámetros (esperado: 9.555.457), época", ck['epoca'])

ds = TormentaDataset(split_dir=f'{DIR_EXT}/val',
                     stats_path=f'{DIR_EXT}/stats.npz')
nombres = sorted(f.name for f in ds.files)
idx = {f.name: i for i, f in enumerate(ds.files)}
print(f'Muestras: {len(nombres)} (esperado: 2271)')


def a_entero(t):
    """Desnormaliza un mapa y lo redondea, como en las métricas."""
    v = np.round(ds.desnormalizar_y(t).numpy())
    assert v.min() >= 0 and v.max() <= 255, (v.min(), v.max())
    return v.astype(np.uint8)


pred = np.zeros((len(nombres), TAM, TAM), dtype=np.uint8)
real = np.zeros_like(pred)
with torch.no_grad():
    for a in tqdm(range(0, len(nombres), LOTE)):
        lote = nombres[a:a + LOTE]
        X = torch.stack([ds[idx[n]][0] for n in lote]).to(device)
        p = m(X).cpu().squeeze(1).clamp(0, 1)
        for k, n in enumerate(lote):
            pred[a + k] = a_entero(p[k])
            real[a + k] = a_entero(ds[idx[n]][1].squeeze())

# Control: CSI promedio sobre los pasos de tiempo (≈ 0,659)
vals = []
for r, e in zip(real, pred):
    act = (r > 0) | (e > 0)
    if act.any():
        vals.append(((r > 0) & (e > 0)).sum() / act.sum())
print(f'CSI promedio: {np.mean(vals):.3f}  (esperado ≈ 0,659)')

os.makedirs('../estudios', exist_ok=True)
np.savez_compressed(SALIDA, archivos=np.array(nombres),
                    y_real=real, pred_ext_ultima=pred)
print(f'Guardado {SALIDA} ({os.path.getsize(SALIDA) / 1e6:.1f} MB)')
