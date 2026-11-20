# -*- coding: utf-8 -*-
"""
Figura del campo receptivo efectivo de los tres modelos.

Lee figuras/datos/campo_efectivo_facl.npz, campo_efectivo_comb.npz y
campo_efectivo.npz (Temp), y genera img/campo_efectivo.pdf:
  (a)-(c) mapa de influencia promedio de cada modelo (escala
          logarítmica común), con el campo teórico (cuadrado negro) y
          los círculos de radio r90 (continuo) y r50 (de trazos);
  (d)     fracción acumulada de la influencia según la distancia al
          píxel de salida, con los tres modelos juntos.

Uso:  python3 figuras/generar_figura_campo_efectivo_tres.py
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif",
                   "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5,
    "axes.linewidth": 0.6, "pdf.fonttype": 42, "savefig.dpi": 300})

ORANGE, GREEN, BLUE, INK = "#eb6834", "#1a9a7a", "#2a78d6", "#222222"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "img")
MODELOS = [("campo_efectivo_facl.npz", "Ext Clip U-Net FACL", ORANGE),
           ("campo_efectivo_comb.npz", "Ext Clip U-Net Comb V2.1",
            GREEN),
           ("campo_efectivo.npz", "Temp Ext Clip U-Net FACL", BLUE)]
D = []
for f, nom, c in MODELOS:
    d = np.load(os.path.join(HERE, "datos", f), allow_pickle=True)
    D.append(dict(nom=nom, c=c, M=d["medio"][0], ac=d["acum"][0],
                  r50=int(d["r50"][0]), r90=int(d["r90"][0]),
                  L=int(d["L_teo"][0])))
H = D[0]["M"].shape[0] // 2
vmax = max(d["M"].max() for d in D)
norm = LogNorm(vmin=vmax * 1e-5, vmax=vmax)
halo = [pe.withStroke(linewidth=2.6, foreground="white")]

fig = plt.figure(figsize=(6.4, 4.7))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.62],
                      width_ratios=[1, 1, 1, 0.05], wspace=0.12,
                      hspace=0.30, left=0.075, right=0.93, top=0.95,
                      bottom=0.085)
ims = []
for k, d in enumerate(D):
    ax = fig.add_subplot(gs[0, k])
    im = ax.imshow(d["M"], extent=(-H - 0.5, H + 0.5, H + 0.5, -H - 0.5),
                   cmap="magma_r", norm=norm)
    ims.append(im)
    L = d["L"]
    ax.add_patch(Rectangle((-L / 2, -L / 2), L, L, fc="none", ec=INK,
                           lw=1.0))
    for r, ls, lw in ((d["r90"], "-", 1.3), (d["r50"], (0, (3, 1.5)),
                                              1.2)):
        ax.add_patch(Circle((0, 0), r, fc="none", ec=d["c"], lw=lw,
                            ls=ls, path_effects=halo))
    ax.set_xlim(-H * 1.15, H * 1.15)
    ax.set_ylim(H * 1.15, -H * 1.15)
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_yticks([-100, -50, 0, 50, 100])
    if k:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel("Distancia vertical (píxeles)")
    ax.set_xlabel("Distancia horizontal (píxeles)", fontsize=7)
    ax.set_title("(" + "abc"[k] + ") " + d["nom"].replace(
        "Ext Clip U-Net", "Ext Clip\nU-Net").replace(
        "Temp Ext Clip U-Net", "Temp Ext Clip\nU-Net"),
        fontsize=7.2, loc="left", linespacing=1.15)
cax = fig.add_subplot(gs[0, 3])
cb = fig.colorbar(ims[0], cax=cax)
cb.set_label("Influencia (fracción por píxel)", fontsize=7)

ax2 = fig.add_subplot(gs[1, :3])
r = np.arange(len(D[0]["ac"]))
for d in D:
    ax2.plot(r, d["ac"], color=d["c"], lw=1.3)
    ax2.axvline(d["L"] / 2, color=d["c"], lw=0.8, ls=":")
    ax2.plot([d["r50"], d["r90"]], [0.5, 0.9], ls="none", marker="o",
             ms=4, mfc="white", mec=d["c"], mew=1.1, zorder=5)
ax2.axhline(0.9, color="#8a8a85", lw=0.6)
ax2.axhline(0.5, color="#8a8a85", lw=0.6)
ax2.set_xlim(0, 120)
ax2.set_ylim(0, 1.02)
ax2.set_xlabel("Distancia al píxel de salida (píxeles)")
ax2.set_ylabel("Fracción acumulada\nde la influencia")
ax2.set_title("(d)", fontsize=7.2, loc="left")
for s_ in ("top", "right"):
    ax2.spines[s_].set_visible(False)
asas = [Line2D([], [], color=d["c"], lw=1.3) for d in D] + [
    Line2D([], [], color="#555555", ls="none", marker="o", ms=4,
           mfc="white", mew=1.1),
    Line2D([], [], color="#555555", lw=0.8, ls=":")]
ax2.legend(asas, [d["nom"] for d in D] + [
    "$r_{50}$ y $r_{90}$", "Radio teórico"], loc="lower right",
    frameon=False, fontsize=7)
os.makedirs(OUT, exist_ok=True)
ruta = os.path.join(OUT, "campo_efectivo.pdf")
fig.savefig(ruta, bbox_inches="tight", pad_inches=0.03)
fig.savefig(ruta.replace(".pdf", ".png"), bbox_inches="tight",
            pad_inches=0.03, dpi=200)
print("OK", ruta)
