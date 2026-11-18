"""
Histograma de LIGHTDENS (Capítulo 5, Rangos y Normalización).

Uso (desde la carpeta de la monografía):
    python3 figuras/generar_figura_hist_lightdens.py

Lee figuras/datos/hist_lightdens.npz, que genera
figuras/celda_hist_lightdens.py desde Datasets.ipynb, y escribe
img/hist_lightdens.pdf.
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif",
                   "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9,
    "axes.linewidth": 0.6, "pdf.fonttype": 42, "savefig.dpi": 300,
})
BLUE = "#2a78d6"
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = "figuras"
d = np.load(os.path.join(HERE, "datos", "hist_lightdens.npz"))
OUT = os.path.join(HERE, "..", "img")
os.makedirs(OUT, exist_ok=True)

fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.7))
for ax, e, c, xl, tit in [
        (axs[0], d["edges_y"], d["counts_y"],
         "LIGHTDENS $y$ (descargas por columna)", "(a)"),
        (axs[1], d["edges_l"], d["counts_l"],
         "$\\ln(y+0{,}1)$", "(b)")]:
    ax.bar(e[:-1], c, width=np.diff(e), align="edge", color=BLUE,
           edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel(xl)
    ax.set_title(tit, loc="left", fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axs[0].set_ylabel("Cantidad de columnas")
axs[1].set_xlim(-0.3, 5.3)   # y >= 1: no hay valores activos por debajo de 0,1
fig.tight_layout()
fig.savefig(os.path.join(OUT, "hist_lightdens.pdf"))
print("img/hist_lightdens.pdf")
