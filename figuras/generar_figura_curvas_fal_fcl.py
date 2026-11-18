"""
Figura de la Sección 5.4.2: evolución de la FAL y la FCL de
validación por separado (reproducción del ensayo 39 de la Fase 2).

Uso (desde la carpeta de la monografía):
    python3 figuras/generar_figura_curvas_fal_fcl.py
Lee figuras/datos/curvas_fal_fcl_trial0039.json y escribe
img/curvas_fal_fcl.pdf (y .png).
"""
import json
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
INK, MUTED, BLUE, ORANGE = "#222222", "#8a8a85", "#2a78d6", "#eb6834"

HERE = os.path.dirname(os.path.abspath(__file__))
h = json.load(open(os.path.join(HERE, "datos",
                                "curvas_fal_fcl_trial0039.json")))["hist"]
ep = np.array(h["epoca"])
fal, fcl = np.array(h["FAL"]), np.array(h["FCL"])
lval = np.array(h["L_val"])
p = np.array(h["prob_fcl"])
e_min = int(ep[lval.argmin()])
e_ult = int(ep[-1])


def coma(x, nd):
    return f"{x:.{nd}f}".replace(".", ",")


fig, axs = plt.subplots(3, 1, figsize=(5.6, 6.4), sharex=True)
for ax, y, col, tit, esc in (
        (axs[0], fcl, BLUE, r"FCL de validación", 1),
        (axs[1], fal * 1e3, ORANGE,
         r"FAL de validación ($\times10^{-3}$)", 1),
        (axs[2], lval, INK, r"$L_{\mathrm{val}}$", 1)):
    ax2 = ax.twinx()
    ax2.fill_between(ep, 0, p, color=MUTED, alpha=0.13, lw=0)
    ax2.set_ylim(0, 1.0)
    ax2.set_yticks([0, 0.5, 1])
    ax2.tick_params(colors=MUTED, labelsize=8)
    ax2.spines["right"].set_color(MUTED)
    ax2.spines["top"].set_visible(False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.plot(ep, y, color=col, lw=1.3)
    if ax is axs[2]:
        ax.plot(ep, 0.5 * fcl, color=BLUE, lw=1.0, ls=(0, (4, 2)))
    for e, ls in ((e_min, "--"), (e_ult, ":")):
        ax.axvline(e, color=INK, lw=0.7, ls=ls)
        ax.plot(e, y[e - 1], "o", color=col, ms=4.5, mec=INK, mew=0.7)
    ax.set_ylabel(tit)
    ax.spines["top"].set_visible(False)
    ax.tick_params(labelsize=8)
axs[2].set_xlabel("Época")
axs[2].set_xlim(0, 141)
axs[0].text(e_min + 4, fcl.min() + 0.0005,
            f"mínimo de $L_{{\\mathrm{{val}}}}$\n(época {e_min})",
            fontsize=7.5, va="bottom", ha="left")
axs[0].text(e_ult - 3, fcl.min() + 0.0005,
            f"última\n(época {e_ult})", fontsize=7.5,
            va="bottom", ha="right")
axs[2].text(62, 0.0475, r"$\frac{1}{2}\,$FCL",
            color=BLUE, fontsize=7.5, va="bottom", ha="center")
fig.text(0.995, 0.5, r"$P(\mathrm{FCL})$ (sombreado)", rotation=90,
         color=MUTED, va="center", ha="right", fontsize=8)
fig.tight_layout(rect=(0, 0, 0.97, 1))
out = os.path.join(HERE, "..", "img", "curvas_fal_fcl.pdf")
fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight",
            pad_inches=0.03, dpi=200)
print("OK", out)
