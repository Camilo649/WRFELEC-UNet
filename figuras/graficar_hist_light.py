# -*- coding: utf-8 -*-
"""Grafica la distribución de LIGHT y LIGHTDENS a partir de
datos/hist_light_lightdens.npz (lo genera celda_hist_light.py).
Uso: python graficar_hist_light.py  ->  hist_light_lightdens.png"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("datos/hist_light_lightdens.npz")
vars_ = [("LIGHT", d["light"]), ("LIGHTDENS", d["lightdens"])]

fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
for j, (nombre, c) in enumerate(vars_):
    x = np.arange(len(c))
    # fila 1: valores 1 a 100, escala log en y, picos marcados
    a = ax[0, j]
    m = (x >= 1) & (x <= 100)
    a.bar(x[m], c[m], width=0.9, color="#4c72b0")
    for k in (20, 40, 60, 80, 100):
        a.axvline(k, color="#c44e52", ls=":", lw=0.8)
    a.set_yscale("log")
    a.set_title(f"{nombre}: valores de 1 a 100")
    a.set_xlabel("descargas por columna")
    a.set_ylabel("columnas (escala log)")
    # fila 2: rango completo, log-log
    b = ax[1, j]
    m = (x >= 1) & (c > 0)
    b.plot(x[m], c[m], ".", ms=3, color="#4c72b0")
    b.set_xscale("log"); b.set_yscale("log")
    b.set_title(f"{nombre}: rango completo (hasta 1.200)")
    b.set_xlabel("descargas por columna (escala log)")
    b.set_ylabel("columnas (escala log)")
fig.suptitle("Distribución de LIGHT y LIGHTDENS en los wrfout "
             f"(1 de cada {int(d['paso'])} salidas, {int(d['n_archivos'])} archivos)")
fig.tight_layout()
fig.savefig("hist_light_lightdens.png", dpi=150)
print("guardado hist_light_lightdens.png")
