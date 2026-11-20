"""
Figura del campo receptivo efectivo (Sección "Campo receptivo
efectivo" de Resultados).

Lee figuras/datos/campo_efectivo.npz, que genera la celda
celda_campo_receptivo_efectivo.py, y genera img/campo_efectivo.pdf:
  (a) mapa de influencia promedio del último modelo de la lista (el
      modelo final), en escala logarítmica, con el campo teórico
      (cuadrado negro) y los círculos de radio r90 (rojo, continuo)
      y r50 (rojo, de trazos);
  (b) fracción acumulada de la influencia según la distancia al píxel
      de salida, con el radio teórico (línea de puntos), el r90
      (línea de trazo) y el r50 (línea de trazo y punto), y las
      líneas horizontales del 50 % y el 90 % de la influencia.

Uso (desde la carpeta de la monografía):
    python3 figuras/generar_figura_campo_efectivo.py
Opcional: pasar otro .npz y otra carpeta de salida:
    python3 figuras/generar_figura_campo_efectivo.py datos.npz salida/
"""
import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Rectangle

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif",
                   "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
})

INK = "#222222"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#c0392b"
GREEN = "#1a9a7a"        # color de la Comb V2.1 en toda la monografía
COLORES = [BLUE, ORANGE, "#3a8a5a", "#8a5ab8"]

HERE = os.path.dirname(os.path.abspath(__file__))
RUTA_NPZ = (sys.argv[1] if len(sys.argv) > 1
            else os.path.join(HERE, "datos", "campo_efectivo.npz"))
OUT = (sys.argv[2] if len(sys.argv) > 2
       else os.path.join(HERE, "..", "img"))
os.makedirs(OUT, exist_ok=True)

D = np.load(RUTA_NPZ, allow_pickle=True)
nombres = [str(n) for n in D["nombres"]]
medio, acum = D["medio"], D["acum"]
r50, r90, L_teo = D["r50"], D["r90"], D["L_teo"]
H = medio.shape[-1] // 2

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(6.4, 2.9),
    gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.55})

# --- (a) mapa de influencia del modelo final ---------------------------
k = len(nombres) - 1
M = medio[k]
vmax = M.max()
im = ax1.imshow(M, extent=(-H - 0.5, H + 0.5, H + 0.5, -H - 0.5),
                cmap="magma_r",
                norm=LogNorm(vmin=max(vmax * 1e-5, M[M > 0].min()),
                             vmax=vmax))
lado = L_teo[k]
ax1.add_patch(Rectangle((-lado / 2, -lado / 2), lado, lado, fc="none",
                        ec=INK, lw=1.0))
ax1.add_patch(Circle((0, 0), r90[k], fc="none", ec=GREEN, lw=1.2))
ax1.add_patch(Circle((0, 0), r50[k], fc="none", ec=GREEN, lw=1.0,
                     ls=(0, (3, 1.5))))
ax1.set_xlim(-H * 1.15, H * 1.15)
ax1.set_ylim(H * 1.15, -H * 1.15)
ax1.set_xlabel("Distancia horizontal al píxel de salida (píxeles)")
ax1.set_ylabel("Distancia vertical (píxeles)")
ax1.set_title(f"(a) {nombres[k]}", fontsize=8, loc="left")
cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
cb.set_label("Influencia (fracción por píxel)", fontsize=7)

# --- (b) influencia acumulada según la distancia -----------------------
from matplotlib.lines import Line2D
radios = np.arange(acum.shape[1])
for i, nom in enumerate(nombres):
    c = GREEN if len(nombres) == 1 else COLORES[i % len(COLORES)]
    ax2.plot(radios, acum[i], color=c, lw=1.2)
    ax2.axvline(L_teo[i] / 2, color=c, lw=0.8, ls=":")
    ax2.axvline(r90[i], color=c, lw=0.8, ls="--")
    ax2.axvline(r50[i], color=c, lw=0.8, ls="-.")
ax2.axhline(0.9, color="#8a8a85", lw=0.6)
ax2.axhline(0.5, color="#8a8a85", lw=0.6)
ax2.set_xlim(0, H * 1.2)
ax2.set_ylim(0, 1.02)
ax2.set_xlabel("Distancia al píxel de salida (píxeles)")
ax2.set_ylabel("Fracción acumulada de la influencia")
ax2.set_title("(b)", fontsize=8, loc="left")
if len(nombres) == 1:       # un solo modelo: leyenda de los trazos
    c = GREEN
    asas = [Line2D([], [], color=c, lw=1.2),
            Line2D([], [], color=c, lw=0.8, ls=":"),
            Line2D([], [], color=c, lw=0.8, ls="--"),
            Line2D([], [], color=c, lw=0.8, ls="-.")]
    ax2.legend(asas, ["Influencia acumulada",
                      f"Radio teórico ({L_teo[0] // 2} px)",
                      f"$r_{{90}}$ ({r90[0]} px)",
                      f"$r_{{50}}$ ({r50[0]} px)"],
               loc="lower right", frameon=False, fontsize=7)
else:                       # varios modelos: un color por modelo
    asas = [Line2D([], [], color=COLORES[i % len(COLORES)], lw=1.2)
            for i in range(len(nombres))]
    ax2.legend(asas, nombres, loc="lower right", frameon=False,
               fontsize=7)
for s_ in ("top", "right"):
    ax2.spines[s_].set_visible(False)

ruta = os.path.join(OUT, "campo_efectivo.pdf")
fig.savefig(ruta, bbox_inches="tight", pad_inches=0.03)
fig.savefig(ruta.replace(".pdf", ".png"), bbox_inches="tight",
            pad_inches=0.03, dpi=200)
plt.close(fig)
print("OK", ruta)
