# -*- coding: utf-8 -*-
"""
Histograma de LIGHT y de LIGHTDENS (sin restringir) de los wrfout, para
ver si LIGHT presenta picos en valores periódicos como LIGHTDENS.

Pegar en una celda de Datasets.ipynb, donde ya existe `eventos`
(celda "Corridas continuas"). Usa un subconjunto de las salidas
(una de cada PASO) para que no demore. Guarda
figuras/datos/hist_light_lightdens.npz y imprime un resumen.
"""
import os
import numpy as np
from netCDF4 import Dataset
from tqdm.auto import tqdm

PASO = 3                        # usar una de cada PASO salidas
VMAX = 1200
archivos = sorted({f for fs in eventos.values() for f in fs})[::PASO]

cuentas = {"LIGHT": np.zeros(VMAX + 2, np.int64),
           "LIGHTDENS": np.zeros(VMAX + 2, np.int64)}
no_enteros = {"LIGHT": 0, "LIGHTDENS": 0}
for f in tqdm(archivos):
    with Dataset(f) as ds:
        for v in cuentas:
            x = np.squeeze(np.asarray(ds.variables[v][:], np.float64))
            x = x[x > 0]
            no_enteros[v] += int((x != np.round(x)).sum())
            r = np.minimum(np.round(x).astype(np.int64), VMAX + 1)
            cuentas[v] += np.bincount(r, minlength=VMAX + 2)

os.makedirs("figuras/datos", exist_ok=True)
np.savez("figuras/datos/hist_light_lightdens.npz",
         light=cuentas["LIGHT"], lightdens=cuentas["LIGHTDENS"],
         paso=PASO, n_archivos=len(archivos))
print(len(archivos), "salidas usadas")
for v, c in cuentas.items():
    tot = c[1:].sum()
    print(f"\n== {v} ==  columnas con actividad: {tot}  "
          f"(valores no enteros: {no_enteros[v]})")
    print("máximo:", int(np.nonzero(c)[0].max()))
    top = np.argsort(c[1:])[::-1][:12] + 1
    print("12 valores más frecuentes:",
          [(int(t), int(c[t])) for t in top])
    m20 = c[20::20].sum()
    print(f"fracción en múltiplos de 20: {m20 / tot:.3f} "
          f"(esperada sin picos, aprox. {len(c[20::20]) / VMAX:.3f} "
          "del rango)")
    for k in (20, 40, 60, 80):
        print(f"  valor {k}: {c[k]} | vecinos {c[k-1]} y {c[k+1]}")
