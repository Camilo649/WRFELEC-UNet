# -*- coding: utf-8 -*-
"""Esquema propio y simplificado de un cumulonimbus (Figura 2.1).

Muestra solo lo que usa el capítulo: corriente ascendente, yunque,
tropopausa, región de fase mixta con granizo, precipitación,
corriente descendente y salida de aire frío.
Salida: img/Cumulonimbus.pdf y img/Cumulonimbus.png
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, FancyArrowPatch
from matplotlib.path import Path
from scipy.interpolate import splprep, splev

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Liberation Serif", "Times New Roman",
                              "DejaVu Serif"]
mpl.rcParams["pdf.fonttype"] = 42

INK = "#2b2b2b"
MUTED = "#6b6b6b"
NUBE = "#eceff3"
BORDE = "#7d8794"
MIXTA = "#7fa6d1"
CALIDO = "#c8553d"
FRIO = "#2f6ca3"

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "..", "img")

# contorno de la nube (sentido horario desde la base izquierda)
pts = np.array([
    (3.30, 2.45), (3.00, 3.60), (3.05, 5.00), (3.35, 6.40),
    (3.40, 7.25), (2.30, 7.60), (0.90, 7.95), (0.85, 8.10),
    (2.20, 8.30), (3.60, 8.45), (4.70, 8.62), (6.00, 8.47),
    (7.60, 8.30), (9.10, 8.12), (9.35, 7.95), (8.00, 7.62),
    (6.80, 7.25), (6.65, 6.40), (6.55, 5.00), (6.75, 3.60),
    (7.10, 2.60), (6.40, 2.20), (4.80, 2.20),
])
pts = np.vstack([pts, pts[:1]])
tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0.03, per=True)
xs, ys = splev(np.linspace(0, 1, 600), tck)
contorno = Path(np.column_stack([xs, ys]))

fig, ax = plt.subplots(figsize=(6.6, 4.5))
ax.set_xlim(0, 10)
ax.set_ylim(0.05, 9.6)
ax.axis("off")

# suelo
ax.plot([0, 10], [0.95, 0.95], color=INK, lw=1.2,
        solid_capstyle="butt")
ax.fill_between([0, 10], 0.70, 0.95, color="#d9d4c7", lw=0)

# nube y región de fase mixta (recortada a la nube)
nube = PathPatch(contorno, fc=NUBE, ec=BORDE, lw=1.1, zorder=2)
ax.add_patch(nube)
banda = ax.fill_between([0, 10], 3.45, 7.0, color=MIXTA, alpha=0.30,
                        lw=0, zorder=3)
banda.set_clip_path(nube)

# niveles de referencia
y0, ytrop = 3.45, 8.95
ax.plot([0.2, 9.8], [y0, y0], ls=(0, (4, 3)), color=MUTED, lw=0.9,
        zorder=1)
ax.plot([0.2, 9.8], [ytrop, ytrop], ls=(0, (4, 3)), color=MUTED,
        lw=0.9, zorder=1)

# granizo (puntos) dentro de la región de fase mixta
rng = np.random.default_rng(3)
gx = rng.uniform(3.9, 5.7, 26)
gy = rng.uniform(3.7, 6.2, 26)
ax.scatter(gx, gy, s=9, color="#4a4a4a", zorder=4, lw=0)

# precipitación bajo la nube (lluvia y granizo)
for x in np.linspace(6.15, 7.15, 6):
    ax.plot([x, x - 0.12], [2.15, 1.35], color=FRIO, lw=0.8, alpha=0.7,
            zorder=3)


def flecha(puntos, color, lw=2.6, ms=14, z=5):
    """Curva suave que termina en una punta de flecha."""
    p = np.array(puntos)
    t, _ = splprep([p[:, 0], p[:, 1]], s=0, k=min(3, len(p) - 1))
    x, y = splev(np.linspace(0, 1, 120), t)
    ax.plot(x[:-3], y[:-3], color=color, lw=lw, zorder=z,
            solid_capstyle="round")
    ax.add_patch(FancyArrowPatch((x[-6], y[-6]), (x[-1], y[-1]),
                                 arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, zorder=z,
                                 shrinkA=0, shrinkB=0))


# corriente ascendente: entrada, ascenso y salida por el yunque
flecha([(0.9, 1.65), (2.4, 1.95), (3.7, 2.9), (4.25, 4.6), (4.4, 6.6),
        (5.0, 7.85), (7.4, 8.05), (9.0, 7.95)], CALIDO)

# corriente descendente y salida de aire frío
flecha([(5.55, 6.1), (5.55, 3.8), (5.5, 2.3), (5.6, 1.5)], FRIO,
       lw=2.2, ms=12)
flecha([(5.6, 1.5), (6.8, 1.25), (8.4, 1.25), (9.4, 1.25)], FRIO,
       lw=2.2, ms=12)


def rotulo(xy_texto, xy_punta, texto, ha="left", va="center", lw=0.6):
    ax.annotate(texto, xy=xy_punta, xytext=xy_texto, ha=ha, va=va,
                fontsize=8.6, color=INK, zorder=6,
                arrowprops=dict(arrowstyle="-", lw=lw, color=MUTED,
                                shrinkA=2, shrinkB=0))


rotulo((0.25, 6.6), (1.55, 7.85), "Yunque")
rotulo((0.25, 5.0), (4.25, 5.0), "Corriente ascendente")
rotulo((0.25, 2.7), (1.6, 1.82), "Entrada de aire\ncálido y húmedo")
rotulo((7.35, 6.0), (6.4, 5.6),
       "Región de fase mixta\n(agua sobreenfriada,\ncristales y granizo)")
rotulo((7.35, 4.3), (5.62, 4.3), "Corriente descendente")
rotulo((7.75, 2.1), (7.15, 1.9), "Precipitación\n(lluvia y granizo)")
rotulo((7.6, 0.35), (7.4, 1.2), "Salida de aire frío",
       ha="center")
ax.text(9.8, ytrop + 0.17, "Tropopausa", ha="right", va="bottom",
        fontsize=8.6, color=INK)
ax.text(9.8, y0 + 0.17, "Isoterma de 0 °C", ha="right", va="bottom",
        fontsize=8.6, color=INK)

os.makedirs(IMG, exist_ok=True)
fig.savefig(os.path.join(IMG, "Cumulonimbus.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(IMG, "Cumulonimbus.png"), dpi=300,
            bbox_inches="tight")
print("OK", IMG)
