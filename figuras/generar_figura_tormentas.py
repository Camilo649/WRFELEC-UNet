"""
Figura del Capítulo 4 (Datos): distribución de la duración de las
tormentas.

Uso (desde la carpeta de la monografía):
    python3 figuras/generar_figura_tormentas.py

Genera img/duracion_tormentas.pdf. La duración de cada tormenta es
su cantidad de instantes activos (sin los hasta 5 previos) por 2 min.
Los valores salen de la celda de tormentas de Datasets.ipynb
(campo n_activos de cada una de las 65 tormentas).
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator, NullFormatter

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif",
                   "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
})
INK, MUTED, BLUE, ORANGE = "#222222", "#8a8a85", "#2a78d6", "#eb6834"

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
    OUT = os.path.join(HERE, "..", "img")
except NameError:
    OUT = "img"
os.makedirs(OUT, exist_ok=True)

# instantes activos de cada una de las 65 tormentas
N_ACTIVOS = [
    936, 697, 424, 2, 1, 5, 425, 705, 244, 78, 1, 4, 1, 1, 303, 1,
    4, 24, 1, 2, 397, 9, 503, 288, 611, 1, 2, 1, 2, 209, 80, 1, 1,
    3, 8, 1, 1, 8, 1, 464, 254, 1, 24, 27, 1, 648, 941, 4, 97, 1,
    478, 1, 1, 2, 122, 689, 668, 31, 9, 42, 2, 10, 100, 248, 199,
]
PASO_MIN = 2
horas = np.array(N_ACTIVOS) * PASO_MIN / 60
assert len(horas) == 65 and sum(N_ACTIVOS) == 11050

# intervalos que se duplican: [2, 4), [4, 8), ... minutos
bordes = PASO_MIN / 60 * 2.0 ** np.arange(0, 11)
cuentas, _ = np.histogram(horas, bins=bordes)
media, mediana = horas.mean(), np.median(horas)


def coma(x, _):
    s = f"{x:g}".replace(".", ",")
    return s


fig, ax = plt.subplots(figsize=(6.2, 2.9))
ax.bar(bordes[:-1], cuentas, width=np.diff(bordes), align="edge",
       color=BLUE, edgecolor="white", linewidth=0.8)
for xv, txt, col, dy in [(mediana, "mediana", INK, 0.97),
                         (media, "promedio", ORANGE, 0.97)]:
    ax.axvline(xv, color=col, lw=1.1, ls="--")
    ax.text(xv * 1.07, ax.get_ylim()[1] * dy,
            f"{txt}: {xv:.1f} h".replace(".", ","), color=col,
            ha="left", va="top", fontsize=8.5)
for x0, x1, c in zip(bordes[:-1], bordes[1:], cuentas):
    if c:
        ax.text(np.sqrt(x0 * x1), c + 0.3, str(c), ha="center",
                va="bottom", fontsize=8, color=INK, zorder=5,
                bbox=dict(fc="white", ec="none", pad=0.6))
ax.set_xscale("log")
ax.set_xlim(bordes[0], bordes[-1])
ax.set_ylim(0, cuentas.max() * 1.25)
ticks = [0.05, 0.1, 0.5, 1, 5, 10, 30]
ax.xaxis.set_major_locator(FixedLocator(ticks))
ax.xaxis.set_major_formatter(FuncFormatter(coma))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("Duración de la tormenta (horas, escala logarítmica)")
ax.set_ylabel("Cantidad de tormentas")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(width=0.6)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "duracion_tormentas.pdf"))
fig.savefig(os.path.join(OUT, "duracion_tormentas.png"), dpi=200)
print(cuentas, media, mediana)
