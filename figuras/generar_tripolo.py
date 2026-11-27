# -*- coding: utf-8 -*-
"""Esquema propio de la estructura tripolar (Figura 2.4).

Réplica simplificada del esquema de la University of Arizona, con
el estilo de las demás figuras (generar_cumulonimbus.py).
Izquierda: colisiones entre granizo y cristales de hielo por
encima y por debajo de la temperatura de inversión. Derecha: las
tres regiones de carga de la nube (A, B y C).
Salida: img/tripolo.pdf y img/tripolo.png
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import (PathPatch, FancyArrowPatch, Circle,
                                RegularPolygon, Ellipse)
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
POS = "#c8553d"      # carga positiva
NEG = "#2f6ca3"      # carga negativa
GOTA = "#8fb3d6"
FLECHA = "#7a5c99"
GRANIZO = "#b9b9b9"

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "..", "img")
FS = 8.6

fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.set_xlim(0, 11.9)
ax.set_ylim(0.0, 9.75)
ax.set_aspect("equal")
ax.axis("off")

Y_INV = 4.85  # altura de la temperatura de inversión


# ------------------------------------------------------------------
# Piezas
def granizo(x, y, carga=None, r=0.36):
    color = {None: GRANIZO, "+": POS, "-": NEG}[carga]
    ax.add_patch(Circle((x, y), r, fc=color, ec=INK, lw=0.8,
                        alpha=0.9 if carga else 1, zorder=5))
    # textura: puntitos dentro de la bola
    rng = np.random.default_rng(int(x * 10 + y * 100))
    ang = rng.uniform(0, 2 * np.pi, 14)
    rad = r * np.sqrt(rng.uniform(0, 0.75, 14))
    ax.scatter(x + rad * np.cos(ang), y + rad * np.sin(ang), s=1.6,
               color="white" if carga else "#6e6e6e", lw=0, zorder=6)


def cristal(x, y, carga=None, r=0.38):
    color = {None: "white", "+": POS, "-": NEG}[carga]
    ax.add_patch(RegularPolygon((x, y), 6, radius=r,
                                orientation=np.pi / 6, fc=color,
                                ec=INK, lw=0.8,
                                alpha=0.85 if carga else 1, zorder=5))


def signo(x, y, s, size=13):
    ax.text(x, y, "+" if s == "+" else "−", color=POS if s == "+"
            else NEG, fontsize=size, fontweight="bold", ha="center",
            va="center", zorder=7)


def letra(x, y, l):
    ax.text(x, y, l, fontsize=FS, fontweight="bold", ha="center",
            va="center", color=INK, zorder=9,
            bbox=dict(boxstyle="circle,pad=0.22", fc="white", ec=INK,
                      lw=0.8))


def flecha(puntos, color=FLECHA, lw=1.6, ms=11):
    p = np.array(puntos)
    ax.plot(p[:-1, 0].tolist() + [p[-1, 0]],
            p[:-1, 1].tolist() + [p[-1, 1]], color=color, lw=lw,
            zorder=4, solid_joinstyle="round")
    ax.add_patch(FancyArrowPatch(tuple(p[-2] + 0.7 * (p[-1] - p[-2])),
                                 tuple(p[-1]), arrowstyle="-|>",
                                 mutation_scale=ms, color=color,
                                 lw=lw, zorder=4, shrinkA=0,
                                 shrinkB=0))


# ------------------------------------------------------------------
# Gotas sobreenfriadas de fondo (panel izquierdo)
rng = np.random.default_rng(7)
gx = rng.uniform(0.2, 4.4, 70)
gy = rng.uniform(0.35, 9.4, 70)
lejos = np.abs(gy - Y_INV) > 0.35
ax.scatter(gx[lejos], gy[lejos], s=5, color=GOTA, lw=0, zorder=2)


def colision(y0, signo_granizo, signo_cristal, l_granizo, l_cristal):
    """Un granizo cae, un cristal sube, chocan y se cargan.

    y0 es la altura de la parte baja del recuadro.
    """
    xg, xc = 1.1, 3.1
    # granizo descargado arriba, cae y queda cargado abajo
    granizo(xg, y0 + 3.45)
    flecha([(xg, y0 + 2.95), (xg, y0 + 1.55)])
    granizo(xg, y0 + 1.05, signo_granizo)
    signo(xg - 0.65, y0 + 1.05, signo_granizo)
    letra(xg - 0.55, y0 + 0.45, l_granizo)
    # cristal descargado abajo, sube rozando el granizo y se carga
    cristal(xc, y0 + 0.85)
    flecha([(xc - 0.1, y0 + 1.3), (xg + 0.65, y0 + 2.3),
            (xc - 0.15, y0 + 3.2)])
    cristal(xc + 0.15, y0 + 3.6, signo_cristal)
    signo(xc - 0.5, y0 + 3.95, signo_cristal)
    letra(xc + 0.85, y0 + 3.85, l_cristal)


colision(Y_INV + 0.3, "-", "+", "B", "A")   # más frío
colision(0.2, "+", "-", "C", "B")           # más cálido

# rótulos del panel izquierdo
ax.text(0.15, 9.25, "Granizo", fontsize=FS, color=INK, va="center")
ax.text(3.1, 0.25, "Cristal de hielo", fontsize=FS, color=INK,
        ha="center", va="center")
ax.annotate("Gotas de agua\nsobreenfriadas", xy=(3.62, 2.42),
            xytext=(3.95, 2.0), fontsize=FS, color=INK, va="center",
            arrowprops=dict(arrowstyle="-", lw=0.6, color=MUTED,
                            shrinkA=2, shrinkB=1))
ax.scatter([3.62], [2.42], s=7, color=GOTA, lw=0, zorder=3)

# ------------------------------------------------------------------
# Nube con las tres regiones (panel derecho)
pts = np.array([
    (7.4, 1.0), (6.95, 1.5), (6.85, 2.3), (7.15, 2.9), (6.85, 3.6),
    (6.85, 4.4), (7.15, 5.0), (6.95, 5.7), (7.0, 6.5), (7.3, 7.1),
    (7.2, 7.7), (6.6, 8.1), (6.25, 8.6), (6.8, 9.05), (8.2, 9.25),
    (9.8, 9.3), (11.0, 9.35), (11.7, 9.4), (11.0, 9.05),
    (10.2, 8.65), (9.65, 8.2), (9.7, 7.5), (9.95, 6.7),
    (9.75, 5.9), (10.0, 5.0), (9.85, 4.2), (10.1, 3.3),
    (9.9, 2.5), (10.15, 1.6), (9.95, 1.0), (8.7, 0.95),
])
pts = np.vstack([pts, pts[:1]])
tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0.01, per=True)
xs, ys = splev(np.linspace(0, 1, 600), tck)
contorno = Path(np.column_stack([xs, ys]))
nube = PathPatch(contorno, fc=NUBE, ec=BORDE, lw=1.1, zorder=2)
ax.add_patch(nube)

Y_AB = 6.55
for y_a, y_b, color in [(Y_AB, 9.6, POS), (Y_INV, Y_AB, NEG)]:
    banda = ax.fill_between([6.0, 11.9], y_a, y_b, color=color,
                            alpha=0.16, lw=0, zorder=3)
    banda.set_clip_path(nube)
reg_c = Ellipse((8.45, 3.95), 1.9, 1.05, fc=POS, alpha=0.16, lw=0,
                zorder=3)
ax.add_patch(reg_c)
reg_c.set_clip_path(nube)


def signos_en(y_min, y_max, s, dx=0.8, dy=0.6, margen=0.3):
    """Reparte signos en una grilla desfasada dentro de la nube."""
    for k, y in enumerate(np.arange(y_min + dy / 2, y_max, dy)):
        for x in np.arange(6.7 + (k % 2) * dx / 2, 11.6, dx):
            dentro = all(contorno.contains_point((x + ex, y + ey))
                         for ex, ey in [(margen, 0), (-margen, 0),
                                        (0, margen), (0, -margen)])
            if dentro:
                signo(x, y, s, size=11)


signos_en(Y_AB + 0.1, 9.2, "+")
signos_en(Y_INV + 0.1, Y_AB - 0.05, "-")
for x, y in [(8.15, 3.95), (8.75, 4.0)]:
    signo(x, y, "+", size=11)
letra(11.05, 8.45, "A")
letra(10.6, 5.75, "B")
letra(10.7, 3.95, "C")

# ------------------------------------------------------------------
# Temperatura de inversión, a lo ancho de toda la figura
ax.plot([0.1, 11.8], [Y_INV, Y_INV], ls=(0, (4, 3)), color=MUTED,
        lw=0.9, zorder=4)
ax.text(4.9, Y_INV + 0.12, "Temperatura de inversión",
        fontsize=FS, color=INK, ha="center", va="bottom", zorder=9,
        bbox=dict(fc="white", ec="none", pad=0.6))

os.makedirs(IMG, exist_ok=True)
fig.savefig(os.path.join(IMG, "tripolo.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(IMG, "tripolo.png"), dpi=300,
            bbox_inches="tight")
print("OK", IMG)
