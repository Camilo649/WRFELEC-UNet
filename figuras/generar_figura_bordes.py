"""
Figura y cifras de la sección sobre la actividad en los bordes del
dominio (capítulo 6).

Uso: python3 figuras/generar_figura_bordes.py
Lee figuras/datos/resultados_validacion.npz y escribe
img/bordes_dominio.pdf; imprime las cifras citadas en el texto.
"""
import datetime as dt
import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from generar_figuras_instante import (  # noqa: E402
    limites, guardar, SEQ_RAYOS, INK, MUTED)

D = np.load(os.path.join(HERE, "datos", "resultados_validacion.npz"))
ARCH = [str(a) for a in D["archivos"]]
T = [dt.datetime.strptime(a[11:30], "%Y-%m-%d_%H-%M-%S")
     for a in ARCH]
R = D["y_real"]
E = D["pred_ext_ultima"]
M = D["pred_temp"]
n, H, W = R.shape
B = 8                                  # ancho de la banda (columnas)

# ---- Cifras ----------------------------------------------------
bandas = {"inferior": (slice(0, B), slice(None)),
          "superior": (slice(H - B, H), slice(None)),
          "izquierdo": (slice(None), slice(0, B)),
          "derecho": (slice(None), slice(W - B, W))}
tot = int(R.sum(dtype=np.int64))
print(f"Actividad simulada total: {tot}")
for k, (sy, sx) in bandas.items():
    r, e, m = (int(X[:, sy, sx].sum(dtype=np.int64))
               for X in (R, E, M))
    print(f"borde {k:9s}: {100 * r / tot:4.1f} % de la "
          f"actividad; est/sim Ext {e / max(r, 1):.2f}, "
          f"Temp {m / max(r, 1):.2f}")
suma = lambda X: int(X[:, B:-B, B:-B].sum(dtype=np.int64))
print("interior: Ext {:.2f}, Temp {:.2f}".format(
    suma(E) / suma(R), suma(M) / suma(R)))

# ---- Figura ------------------------------------------------------
PASO = "2019-03-04 02:00"
k = [i for i, t in enumerate(T)
     if t.strftime("%Y-%m-%d %H:%M") == PASO][0]
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.7),
                        gridspec_kw={"wspace": 0.05})
for ax, mat, tit in zip(axs, (R[k], E[k], M[k]),
                        ("WRF-ELEC", "Ext Clip U-Net FACL",
                         "Temp Ext Clip U-Net FACL")):
    im = ax.imshow(mat, origin="lower", cmap=SEQ_RAYOS, vmin=0,
                   vmax=150, interpolation="nearest")
    limites(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(tit, fontsize=8, color=INK, pad=3)
    for lado in ax.spines.values():
        lado.set_color(MUTED)
        lado.set_linewidth(0.5)
    ax.add_patch(mpl.patches.Rectangle(
        (B - 0.5, B - 0.5), W - 2 * B, H - 2 * B, fill=False,
        ec="#444444", lw=0.6, ls="--"))
cb = fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.05,
                  pad=0.05, aspect=40, shrink=0.6)
cb.ax.tick_params(labelsize=7.5, length=2.5, width=0.4,
                  labelcolor=MUTED)
cb.outline.set_visible(False)
cb.set_label("LIGHTDENS", fontsize=8, color=MUTED, labelpad=2)
guardar(fig, "bordes_dominio.pdf")
print("paso:", PASO, "borde: sim",
      int(R[k][:B].sum()), "Ext", int(E[k][:B].sum()), "Temp",
      int(M[k][:B].sum()))
