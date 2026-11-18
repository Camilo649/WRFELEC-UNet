# -*- coding: utf-8 -*-
"""
Figura de los errores de los dos modelos columna por columna.

Uso (desde la carpeta de la monografía):
    python3 figuras/generar_figura_errores.py

Lee figuras/datos/resultados_validacion.npz y genera
    img/errores_modelos.pdf
El error de cada modelo es la estimación menos el valor simulado,
en las columnas (de todos los pasos de tiempo de validación) con
actividad simulada o estimada por alguno de los dos modelos. Se
muestra la cantidad de columnas en cada celda de 1 x 1 descargas
y se calcula la correlación de Pearson entre ambos errores.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from generar_figuras_instante import (  # noqa: E402
    guardar, coma, SEQ, INK, MUTED)

D = np.load(os.path.join(HERE, "datos", "resultados_validacion.npz"))
Y, E, T = D["y_real"], D["pred_ext_ultima"], D["pred_temp"]
M = 150                     # máximo de LIGHTDENS
H = np.zeros((2 * M + 1, 2 * M + 1), np.int64)
s = [0, 0, 0, 0, 0]         # n, sum a, sum b, sum a2 / b2 / ab
sa = sb = saa = sbb = sab = 0.0
n = 0
for k in range(len(Y)):
    y = Y[k].astype(np.int32)
    e = E[k].astype(np.int32)
    t = T[k].astype(np.int32)
    u = (y > 0) | (e > 0) | (t > 0)
    a, b = (e - y)[u], (t - y)[u]
    H += np.bincount((a + M) * (2 * M + 1) + (b + M),
                     minlength=(2 * M + 1) ** 2).reshape(
                         2 * M + 1, 2 * M + 1)
    n += a.size
    sa += a.sum(); sb += b.sum()
    saa += (a * a.astype(np.int64)).sum()
    sbb += (b * b.astype(np.int64)).sum()
    sab += (a * b.astype(np.int64)).sum()
r = (sab / n - sa * sb / n ** 2) / np.sqrt(
    (saa / n - (sa / n) ** 2) * (sbb / n - (sb / n) ** 2))
print(f"{n} columnas, correlación de los errores: {r:.3f}")

fig, ax = plt.subplots(figsize=(3.9, 3.6))
# H[i, j]: i = error de la Ext, j = error de la Temp
im = ax.imshow(H.T, origin="lower", cmap=SEQ,
               norm=LogNorm(vmin=1, vmax=H.max()),
               extent=(-M - 0.5, M + 0.5, -M - 0.5, M + 0.5),
               interpolation="nearest")
ax.plot([-M, M], [-M, M], color=MUTED, lw=0.6, ls="--", zorder=3)
for lado in ("top", "right"):
    ax.spines[lado].set_visible(False)
for lado in ("left", "bottom"):
    ax.spines[lado].set_color(MUTED)
ax.tick_params(labelsize=7, length=2.5, width=0.5,
               labelcolor=MUTED, color=MUTED)
ax.set_xlabel("Error de la Ext Clip U-Net FACL", fontsize=7.5,
              color=MUTED)
ax.set_ylabel("Error de la Temp Ext Clip U-Net FACL", fontsize=7.5,
              color=MUTED)
ax.set_xticks(range(-150, 151, 50))
ax.set_yticks(range(-150, 151, 50))
ax.set_xticklabels([str(v).replace("-", "−")
                    for v in range(-150, 151, 50)])
ax.set_yticklabels([str(v).replace("-", "−")
                    for v in range(-150, 151, 50)])
ax.set_aspect("equal")
ax.text(0.04, 0.96, f"$r$ = {coma(r, 2)}", transform=ax.transAxes,
        fontsize=7.5, color=INK, va="top")
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.outline.set_linewidth(0.4)
cb.ax.tick_params(labelsize=6.5, length=2, width=0.4,
                  labelcolor=MUTED, color=MUTED)
cb.set_label("Cantidad de columnas", fontsize=7.5, color=MUTED)
guardar(fig, "errores_modelos.pdf")
