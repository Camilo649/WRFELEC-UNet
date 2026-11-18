"""
Figuras genéricas de la sección 2.3 (Redes Neuronales) de la
monografía. No usan datos del trabajo: son esquemas didácticos con
datos sintéticos (o ejemplos numéricos calculados al vuelo).

Uso (desde la carpeta de la monografía):
    python3 figuras/generar_figuras_redes.py

Genera, en img/:
    redes_neurona_mlp.pdf          (neurona artificial y red MLP)
    redes_descenso_gradiente.pdf   (descenso por el gradiente)
    redes_cnn_operaciones.pdf      (convolución, relleno, paso y
                                    max pooling con ejemplos numéricos)
    redes_campo_receptivo.pdf      (crecimiento del campo receptivo al
                                    apilar convoluciones de 3x3)
    redes_activaciones.pdf         (sigmoide y ReLU, con derivadas)
    redes_sobreajuste.pdf          (subajuste / sobreajuste)
La figura de la U-Net se genera con figuras/unet_diagram.py.
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

# ----------------------------------------------------------------
# Estilo común (tipografía serif, a tono con el texto de la tesis)
# ----------------------------------------------------------------
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

INK = "#222222"        # texto y trazos
MUTED = "#8a8a85"      # elementos secundarios
BLUE = "#2a78d6"       # serie 1
ORANGE = "#eb6834"     # serie 2 / resaltado
LBLUE = "#d6e4f5"      # relleno suave
LORANGE = "#fbdccd"
LGRAY = "#ececea"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "img")
os.makedirs(OUT, exist_ok=True)


def guardar(fig, nombre):
    ruta = os.path.join(OUT, nombre)
    fig.savefig(ruta, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(ruta.replace(".pdf", ".png"), bbox_inches="tight",
                pad_inches=0.03, dpi=200)
    plt.close(fig)
    print("OK", ruta)


def flecha(ax, p, q, color=INK, lw=0.7, ms=6, **kw):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=ms, lw=lw,
        color=color, shrinkA=0, shrinkB=0, **kw))


# ================================================================
# 1) Neurona artificial y red multicapa
# ================================================================
def fig_neurona_mlp():
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(6.3, 2.55),
        gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.05})

    # ---- (a) neurona -------------------------------------------
    a.set_xlim(-0.9, 7.6)
    a.set_ylim(-2.1, 3.1)
    a.set_aspect("equal")
    a.set_anchor("N")
    a.axis("off")
    ys = [2.0, 1.0, 0.0]
    for i, y in enumerate(ys, start=1):
        a.add_patch(Circle((0, y), 0.28, fc=LGRAY, ec=INK, lw=0.6))
        a.text(0, y, rf"$x_{i}$", ha="center", va="center")
        y_fin = 1.0 + (y - 1.0) * 0.25
        flecha(a, (0.30, y), (3.05, y_fin))
        # etiqueta del peso, sobre (o bajo) la flecha
        y_lin = y + (y_fin - y) * (1.3 - 0.3) / 2.75
        dy = 0.22 if y >= 1.0 else -0.24
        a.text(1.3, y_lin + dy, rf"$w_{i}$", ha="center",
               va="center", color=BLUE)
    # sesgo
    a.add_patch(Circle((3.35, -0.9), 0.28, fc=LGRAY, ec=INK, lw=0.6))
    a.text(3.35, -0.9, r"$b$", ha="center", va="center")
    flecha(a, (3.35, -0.6), (3.35, 0.72))
    # suma
    a.add_patch(Circle((3.35, 1.0), 0.68, fc=LBLUE, ec=INK, lw=0.7))
    a.text(3.35, 1.0, r"$\Sigma$", ha="center", va="center",
           fontsize=12)
    flecha(a, (4.03, 1.0), (4.75, 1.0))
    a.text(4.4, 1.42, r"$z$", ha="center", va="center")
    # activación
    a.add_patch(Rectangle((4.75, 0.45), 1.1, 1.1, fc=LORANGE,
                          ec=INK, lw=0.7))
    a.text(5.3, 1.0, r"$f$", ha="center", va="center", fontsize=11)
    flecha(a, (5.85, 1.0), (6.75, 1.0))
    a.text(7.05, 1.0, r"$\hat{y}$", ha="center", va="center", fontsize=9)
    a.text(3.35, 2.35, "suma\nponderada", ha="center",
           va="center", color=MUTED)
    a.text(5.3, 2.35, "función de\nactivación", ha="center",
           va="center", color=MUTED)
    a.text(3.4, -1.75,
           r"$\hat{y}=f\,(w_1x_1+w_2x_2+w_3x_3+b)$",
           ha="center", va="center", fontsize=8)

    a.text(0.0, 1.0, "(a)", transform=a.transAxes, fontsize=8,
           fontweight="bold", va="top", ha="left")

    # ---- (b) red multicapa -------------------------------------
    b.set_xlim(-0.8, 6.4)
    b.set_ylim(-2.7, 3.1)
    b.set_aspect("equal")
    b.set_anchor("N")
    b.axis("off")
    capas = [4, 5, 5, 1]
    xs = [0.0, 2.0, 4.0, 5.9]
    sep = 0.95
    pos = []
    for n, x in zip(capas, xs):
        top = (n - 1) * sep / 2
        pos.append([(x, top - k * sep) for k in range(n)])
    for L in range(len(capas) - 1):
        for p in pos[L]:
            for q in pos[L + 1]:
                b.plot([p[0], q[0]], [p[1], q[1]], color=MUTED,
                       lw=0.3, zorder=1)
    cols = [LGRAY, LBLUE, LBLUE, LORANGE]
    for L, pts in enumerate(pos):
        for (x, y) in pts:
            b.add_patch(Circle((x, y), 0.25, fc=cols[L], ec=INK,
                               lw=0.6, zorder=3))
    for k, (x, y) in enumerate(pos[0], start=1):
        b.text(x, y, rf"$x_{k}$", ha="center", va="center",
               zorder=4, fontsize=7)
    b.text(pos[3][0][0], pos[3][0][1], r"$\hat{y}$", ha="center",
           va="center", zorder=4, fontsize=8)
    for x, t in zip([0.0, 3.0, 5.9],
                    ["capa de\nentrada", "capas ocultas",
                     "capa de\nsalida"]):
        b.text(x, -2.25, t, ha="center", va="top", color=MUTED)
    b.text(0.0, 1.0, "(b)", transform=b.transAxes, fontsize=8,
           fontweight="bold", va="top", ha="left")
    guardar(fig, "redes_neurona_mlp.pdf")


def fig_neurona():
    fig, a = plt.subplots(figsize=(3.3, 2.0))
    # ---- (a) neurona -------------------------------------------
    a.set_xlim(-0.9, 7.6)
    a.set_ylim(-2.1, 3.1)
    a.set_aspect("equal")
    a.set_anchor("N")
    a.axis("off")
    ys = [2.0, 1.0, 0.0]
    for i, y in enumerate(ys, start=1):
        a.add_patch(Circle((0, y), 0.28, fc=LGRAY, ec=INK, lw=0.6))
        a.text(0, y, rf"$x_{i}$", ha="center", va="center")
        y_fin = 1.0 + (y - 1.0) * 0.25
        flecha(a, (0.30, y), (3.05, y_fin))
        # etiqueta del peso, sobre (o bajo) la flecha
        y_lin = y + (y_fin - y) * (1.3 - 0.3) / 2.75
        dy = 0.22 if y >= 1.0 else -0.24
        a.text(1.3, y_lin + dy, rf"$w_{i}$", ha="center",
               va="center", color=BLUE)
    # sesgo
    a.add_patch(Circle((3.35, -0.9), 0.28, fc=LGRAY, ec=INK, lw=0.6))
    a.text(3.35, -0.9, r"$b$", ha="center", va="center")
    flecha(a, (3.35, -0.6), (3.35, 0.72))
    # suma
    a.add_patch(Circle((3.35, 1.0), 0.68, fc=LBLUE, ec=INK, lw=0.7))
    a.text(3.35, 1.0, r"$\Sigma$", ha="center", va="center",
           fontsize=12)
    flecha(a, (4.03, 1.0), (4.75, 1.0))
    a.text(4.4, 1.42, r"$z$", ha="center", va="center")
    # activación
    a.add_patch(Rectangle((4.75, 0.45), 1.1, 1.1, fc=LORANGE,
                          ec=INK, lw=0.7))
    a.text(5.3, 1.0, r"$f$", ha="center", va="center", fontsize=11)
    flecha(a, (5.85, 1.0), (6.75, 1.0))
    a.text(7.05, 1.0, r"$\hat{y}$", ha="center", va="center", fontsize=9)
    a.text(3.35, 2.35, "suma\nponderada", ha="center",
           va="center", color=MUTED)
    a.text(5.3, 2.35, "función de\nactivación", ha="center",
           va="center", color=MUTED)
    a.text(3.4, -1.75,
           r"$\hat{y}=f\,(w_1x_1+w_2x_2+w_3x_3+b)$",
           ha="center", va="center", fontsize=8)

    guardar(fig, "redes_neurona.pdf")


def fig_mlp():
    fig, b = plt.subplots(figsize=(3.4, 2.4))
    # ---- (b) red multicapa -------------------------------------
    b.set_xlim(-0.8, 6.4)
    b.set_ylim(-2.7, 3.1)
    b.set_aspect("equal")
    b.set_anchor("N")
    b.axis("off")
    capas = [4, 5, 5, 1]
    xs = [0.0, 2.0, 4.0, 5.9]
    sep = 0.95
    pos = []
    for n, x in zip(capas, xs):
        top = (n - 1) * sep / 2
        pos.append([(x, top - k * sep) for k in range(n)])
    for L in range(len(capas) - 1):
        for p in pos[L]:
            for q in pos[L + 1]:
                b.plot([p[0], q[0]], [p[1], q[1]], color=MUTED,
                       lw=0.3, zorder=1)
    cols = [LGRAY, LBLUE, LBLUE, LORANGE]
    for L, pts in enumerate(pos):
        for (x, y) in pts:
            b.add_patch(Circle((x, y), 0.25, fc=cols[L], ec=INK,
                               lw=0.6, zorder=3))
    for k, (x, y) in enumerate(pos[0], start=1):
        b.text(x, y, rf"$x_{k}$", ha="center", va="center",
               zorder=4, fontsize=7)
    b.text(pos[3][0][0], pos[3][0][1], r"$\hat{y}$", ha="center",
           va="center", zorder=4, fontsize=8)
    for x, t in zip([0.0, 3.0, 5.9],
                    ["capa de\nentrada", "capas ocultas",
                     "capa de\nsalida"]):
        b.text(x, -2.25, t, ha="center", va="top", color=MUTED)
    guardar(fig, "redes_mlp.pdf")


# ================================================================
# 1b) Funciones de activación (con sus derivadas)
# ================================================================
def fig_activaciones():
    z = np.linspace(-6, 6, 600)
    sig = 1 / (1 + np.exp(-z))
    relu = np.maximum(0, z)
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(4.9, 1.9),
        gridspec_kw={"wspace": 0.28})
    for ax in (a, b):
        ax.axhline(0, color=MUTED, lw=0.5)
        ax.axvline(0, color=MUTED, lw=0.5)
        ax.set_xlabel(r"$z$")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    a.plot(z, sig, color=BLUE, lw=1.3, label=r"$\sigma(z)$")
    a.plot(z, sig * (1 - sig), color=ORANGE, lw=1.1, ls="--",
           label=r"derivada")
    a.set_title(r"Sigmoide: $\sigma(z)=1/(1+e^{-z})$", fontsize=7.5)
    a.set_ylim(-0.1, 1.1)
    a.legend(frameon=False, fontsize=6.8, loc="center right",
             bbox_to_anchor=(1.0, 0.38))
    b.plot(z, relu, color=BLUE, lw=1.3, label=r"$f(z)$")
    b.plot(z[z < 0], np.zeros((z < 0).sum()), color=ORANGE, lw=1.1,
           ls="--")
    b.plot(z[z > 0], np.ones((z > 0).sum()), color=ORANGE, lw=1.1,
           ls="--", label="derivada")
    b.set_title(r"ReLU: $f(z)=\max(0,z)$", fontsize=7.5)
    b.set_ylim(-0.6, 6.4)
    b.legend(frameon=False, fontsize=6.8, loc="upper left")
    guardar(fig, "redes_activaciones.pdf")


# ================================================================
# 2) Descenso por el gradiente
# ================================================================
def fig_descenso():
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(6.3, 2.7),
        gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.28})

    # ---- (a) caso de un parámetro ------------------------------
    L = lambda w: 0.5 * (w - 2.0) ** 2 + 0.3
    dL = lambda w: (w - 2.0)
    w = np.linspace(-1.6, 5.6, 300)
    a.plot(w, L(w), color=INK, lw=1.0)
    eta, w_k = 0.3, -1.0
    ws = [w_k]
    for _ in range(6):
        w_k = w_k - eta * dL(w_k)
        ws.append(w_k)
    ws = np.array(ws)
    for k in range(len(ws) - 1):
        seg = np.linspace(ws[k], ws[k + 1], 30)
        a.plot(seg, L(seg), color=BLUE, lw=1.6, zorder=3)
        # punta de flecha centrada en el tramo (siempre visible)
        xm = ws[k] + 0.62 * (ws[k + 1] - ws[k])
        x0 = ws[k] + 0.50 * (ws[k + 1] - ws[k])
        if k < 4:
            a.annotate("", xy=(xm, L(xm)), xytext=(x0, L(x0)),
                       arrowprops=dict(
                           arrowstyle="-|>,head_length=0.9,"
                                      "head_width=0.32",
                           color=BLUE, lw=0.8, mutation_scale=10,
                           shrinkA=0, shrinkB=0),
                       zorder=4)
    a.plot(ws, L(ws), "o", color=BLUE, ms=3.6, zorder=5)
    for k in range(4):
        dx, dy, ha = (-8, 4, "right") if k == 0 else (7, 6, "left")
        a.annotate(rf"$w_{k}$", (ws[k], L(ws[k])),
                   xytext=(dx, dy), textcoords="offset points",
                   fontsize=7.5, color=INK, ha=ha)
    # tangente en w0
    t = np.linspace(-1.6, 1.0, 20)
    a.plot(t, L(ws[0]) + dL(ws[0]) * (t - ws[0]), "--", color=ORANGE,
           lw=0.8)
    a.text(0.75, 5.2, "pendiente $=$\n$dL/dw$ en $w_0$",
           color=ORANGE, fontsize=7, ha="left", va="center")
    a.axvline(2.0, color=MUTED, lw=0.5, ls=":")
    a.text(2.05, 7.0, "mínimo", color=MUTED, fontsize=7, ha="left")
    a.set_xlabel(r"parámetro $w$")
    a.set_ylabel(r"pérdida $L(w)$")
    a.set_xticks([])
    a.set_yticks([])
    a.set_ylim(0, 8)
    a.text(0.0, 1.02, "(a)", transform=a.transAxes, fontsize=8,
           fontweight="bold", va="bottom")
    a.text(2.6, 2.3, r"$w_{k+1}=w_k-\eta\,\frac{dL}{dw}$",
           fontsize=8.5, ha="left", color=BLUE)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

    # ---- (b) dos parámetros: curvas de nivel -------------------
    c1, c2 = 1.0, 8.0     # curvatura distinta en cada dirección
    L2 = lambda x, y: 0.5 * (c1 * x ** 2 + c2 * y ** 2)
    g2 = lambda p: np.array([c1 * p[0], c2 * p[1]])
    xx, yy = np.meshgrid(np.linspace(-4.6, 4.6, 400),
                         np.linspace(-1.9, 1.9, 400))
    b.contour(xx, yy, L2(xx, yy), levels=[0.2, 0.6, 1.4, 2.8, 4.6,
                                          6.8], colors=MUTED,
              linewidths=0.5)

    def camino(eta, n, p0=(-3.2, 1.0)):
        p = np.array(p0, float)
        pts = [p.copy()]
        for _ in range(n):
            p = p - eta * g2(p)
            pts.append(p.copy())
        return np.array(pts)

    chico = camino(0.04, 40)
    grande = camino(0.22, 14)
    b.plot(*chico.T, "-o", color=BLUE, lw=0.8, ms=1.8,
           label=r"$\eta$ pequeña: lento, sin oscilar", zorder=3)
    b.plot(*grande.T, "-o", color=ORANGE, lw=0.8, ms=2.2,
           label=r"$\eta$ grande: oscila", zorder=4)
    b.plot([-3.2], [1.0], "s", color=INK, ms=3.5, zorder=5)
    b.text(-3.15, 1.12, "inicio", fontsize=7, ha="left",
           va="bottom")
    b.plot([0], [0], "*", color=INK, ms=6, zorder=5)
    b.text(0.15, -0.28, "mínimo", fontsize=7, ha="left")
    b.set_xlabel(r"parámetro $w_1$")
    b.set_ylabel(r"parámetro $w_2$")
    b.set_xticks([])
    b.set_yticks([])
    b.set_xlim(-3.95, 3.95)
    b.set_ylim(-1.5, 1.7)
    b.legend(loc="lower right", frameon=False, fontsize=6.8,
             handlelength=1.6, borderaxespad=0.1)
    b.text(0.0, 1.02, "(b)", transform=b.transAxes, fontsize=8,
           fontweight="bold", va="bottom")
    guardar(fig, "redes_descenso_gradiente.pdf")


# ================================================================
# 3) Convolución (ejemplo numérico)
# ================================================================
def dibujar_grilla(ax, M, x0, y0, celda=1.0, rellenos=None,
                   bordes=None, fs=8, lw=0.6, bold=None):
    n, m = M.shape
    for i in range(n):
        for j in range(m):
            fc = "white" if rellenos is None else rellenos[i][j]
            ec = INK if bordes is None or bordes[i][j] is None \
                else bordes[i][j]
            ax.add_patch(Rectangle((x0 + j * celda,
                                    y0 - (i + 1) * celda),
                                   celda, celda, fc=fc, ec=ec,
                                   lw=lw))
            peso = "bold" if (bold and bold[i][j]) else "normal"
            ax.text(x0 + (j + 0.5) * celda, y0 - (i + 0.5) * celda,
                    f"{M[i, j]:g}", ha="center", va="center",
                    fontsize=fs, fontweight=peso)


def correlacion_valida(X, K):
    n = X.shape[0] - K.shape[0] + 1
    m = X.shape[1] - K.shape[1] + 1
    S = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            S[i, j] = np.sum(X[i:i + K.shape[0], j:j + K.shape[1]]
                             * K)
    return S


def tinte(v, vmax, color=(0.165, 0.471, 0.839)):
    """Mezcla blanco -> azul según v/vmax."""
    t = 0 if vmax == 0 else min(max(v / vmax, 0), 1) * 0.55
    return tuple(1 - t * (1 - c) for c in color)


def _panel(ax, letra, xmax=14.3, ymin=0.0, ymax=9.2):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.3, xmax)
    ax.set_ylim(ymin, ymax)
    ax.text(-0.3, ymax, letra, fontsize=9, fontweight="bold",
            ha="left", va="top")


def _rect(ax, x, y, w, h, color, lw=1.6, ls="-", z=6):
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=color,
                           lw=lw, ls=ls, zorder=z))


def _etiqueta(ax, x, y, titulo, sub=None):
    """Título del bloque (y subtítulo gris debajo del bloque)."""
    ax.text(x, y, titulo, ha="center", va="bottom", fontsize=7.5)


def _panel_convolucion(ax, X, K, top=6.9):
    S = correlacion_valida(X, K)
    fi, fj = 0, 1
    ventana = X[fi:fi + 3, fj:fj + 3]
    assert float(np.sum(ventana * K)) == S[fi, fj]
    rel = [[tinte(X[i, j], 2) for j in range(5)] for i in range(5)]
    dibujar_grilla(ax, X, 0, top, rellenos=rel,
                   bordes=[[None] * 5 for _ in range(5)])
    _rect(ax, fj, top - fi - 3, 3, 3, ORANGE)
    _etiqueta(ax, 2.5, top + 0.25, "entrada")
    ax.text(5.65, top - 2.5, r"$\ast$", ha="center", va="center",
            fontsize=13)
    ky = top - 1.0
    dibujar_grilla(ax, K, 6.2, ky, rellenos=[[LGRAY] * 3] * 3)
    _etiqueta(ax, 7.7, ky + 0.25, "filtro")
    ax.text(7.7, ky - 3.25, "(kernel)", ha="center", va="top",
            color=MUTED, fontsize=6.3)
    ax.text(10.05, ky - 1.5, "$=$", ha="center", va="center",
            fontsize=13)
    relo = [[tinte(S[i, j], 6) for j in range(3)] for i in range(3)]
    brd = [[None] * 3 for _ in range(3)]
    brd[fi][fj] = ORANGE
    dibujar_grilla(ax, S, 10.5, ky, rellenos=relo, bordes=brd,
                   bold=[[(i, j) == (fi, fj) for j in range(3)]
                         for i in range(3)])
    _rect(ax, 10.5 + fj, ky - fi - 1, 1, 1, ORANGE)
    _etiqueta(ax, 12.0, ky + 0.25, "salida")
    ax.text(12.0, ky - 3.25, "(mapa de\ncaracterísticas)",
            ha="center", va="top", color=MUTED, fontsize=6.3,
            linespacing=1.0)


def _panel_padding(ax, X, K, top=7.4):
    P = np.pad(X, 1)                       # relleno de ceros
    S = correlacion_valida(P, K)           # 5x5: conserva el tamaño
    assert S.shape == X.shape
    borde = "#fdf0b8"   # amarillo suave: ceros del relleno
    rel = [[borde if (i in (0, 6) or j in (0, 6))
            else tinte(P[i, j], 2) for j in range(7)]
           for i in range(7)]
    dibujar_grilla(ax, P, 0, top, rellenos=rel,
                   bordes=[[None] * 7 for _ in range(7)])
    _rect(ax, 1, top - 6, 5, 5, INK, lw=1.0, z=5)   # entrada original
    _rect(ax, 0, top - 3, 3, 3, ORANGE)              # ventana (0,0)
    _etiqueta(ax, 3.5, top + 0.25, "entrada con relleno")
    flecha(ax, (7.3, top - 3.5), (8.3, top - 3.5), ms=6)
    oy = top - 1.0
    relo = [[tinte(abs(S[i, j]), 6) for j in range(5)]
            for i in range(5)]
    brd = [[None] * 5 for _ in range(5)]
    brd[0][0] = ORANGE
    dibujar_grilla(ax, S, 8.6, oy, rellenos=relo, bordes=brd,
                   bold=[[(i, j) == (0, 0) for j in range(5)]
                         for i in range(5)])
    _rect(ax, 8.6, oy - 1, 1, 1, ORANGE)
    _etiqueta(ax, 11.1, top + 0.25, "salida")


def _panel_stride(ax, X, K, top=6.9):
    S = np.array([[np.sum(X[i:i + 3, j:j + 3] * K)
                   for j in (0, 2)] for i in (0, 2)])
    rel = [[tinte(X[i, j], 2) for j in range(5)] for i in range(5)]
    dibujar_grilla(ax, X, 0, top, rellenos=rel,
                   bordes=[[None] * 5 for _ in range(5)])
    azul2 = "#1f5fb0"
    _rect(ax, 0, top - 3, 3, 3, ORANGE)
    _rect(ax, 2, top - 3, 3, 3, azul2, ls=(0, (3, 2)))
    _etiqueta(ax, 2.5, top + 0.25, "entrada")
    ax.annotate("", xy=(3.5, top - 5.6), xytext=(1.5, top - 5.6),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=INK,
                                mutation_scale=7))
    ax.text(2.5, top - 5.9, "paso $=2$", ha="center", va="top",
            fontsize=7)
    flecha(ax, (5.7, top - 2.5), (6.9, top - 2.5), ms=6)
    oy = top - 1.5
    relo = [[tinte(abs(S[i, j]), 6) for j in range(2)]
            for i in range(2)]
    dibujar_grilla(ax, S, 7.4, oy, rellenos=relo,
                   bordes=[[ORANGE, azul2], [None, None]])
    _rect(ax, 7.4, oy - 1, 1, 1, ORANGE)
    _rect(ax, 8.4, oy - 1, 1, 1, azul2, ls=(0, (3, 2)))
    _etiqueta(ax, 8.4, oy + 0.25, "salida")


def _panel_pooling(ax, top=6.9, dx=2.6):
    X = np.array([[1, 3, 2, 1],
                  [4, 6, 5, 2],
                  [1, 2, 9, 3],
                  [0, 5, 4, 8]], float)
    S = np.array([[X[:2, :2].max(), X[:2, 2:].max()],
                  [X[2:, :2].max(), X[2:, 2:].max()]])
    tintes = ["#d6e4f5", "#fbdccd", "#d9efe3", "#e6e0f7"]
    rel = [[tintes[(i // 2) * 2 + (j // 2)] for j in range(4)]
           for i in range(4)]
    bold = [[X[i, j] == S[i // 2, j // 2] for j in range(4)]
            for i in range(4)]
    dibujar_grilla(ax, X, dx, top, rellenos=rel, bold=bold)
    ax.plot([dx + 2, dx + 2], [top - 4, top], color=INK, lw=1.3)
    ax.plot([dx, dx + 4], [top - 2, top - 2], color=INK, lw=1.3)
    _etiqueta(ax, dx + 2, top + 0.25, "entrada")
    flecha(ax, (dx + 4.3, top - 2), (dx + 5.15, top - 2), ms=6)
    ax.text(dx + 4.72, top - 1.55, "máx.", ha="center", va="bottom",
            fontsize=6.3, color=MUTED)
    rel_o = [[tintes[i * 2 + j] for j in range(2)] for i in range(2)]
    dibujar_grilla(ax, S, dx + 5.3, top - 1, rellenos=rel_o)
    _etiqueta(ax, dx + 6.3, top + 0.25, "salida")


def fig_operaciones_cnn():
    """Convolución, relleno, paso y max pooling en una sola figura."""
    X = np.array([[2, 2, 2, 0, 0]] * 5, float)
    K = np.array([[1, 0, -1]] * 3, float)
    fig, axs = plt.subplots(2, 2, figsize=(6.3, 4.3),
                            gridspec_kw={"hspace": 0.05,
                                         "wspace": 0.04})
    for ax, let in zip(axs.flat, "abcd"):
        _panel(ax, f"({let})")
    _panel_convolucion(axs[0, 0], X, K)
    _panel_padding(axs[0, 1], X, K)
    _panel_stride(axs[1, 0], X, K)
    _panel_pooling(axs[1, 1])
    guardar(fig, "redes_cnn_operaciones.pdf")


def fig_campo_receptivo_capas():
    """Campo receptivo de un píxel tras apilar tres convoluciones de
    3x3 (con relleno): 1x1 en la capa 3, 3x3 en la 2, 5x5 en la 1 y
    7x7 en la entrada."""
    n, sep = 9, 2.4
    titulos = ["Entrada", "Capa 1", "Capa 2", "Capa 3"]
    lados = [7, 5, 3, 1]
    c0 = (n - 1) // 2
    fig, ax = plt.subplots(figsize=(6.3, 1.95))
    ax.set_aspect("equal")
    ax.axis("off")
    cajas = []
    for k, (tit, lado) in enumerate(zip(titulos, lados)):
        x0 = k * (n + sep)
        i0 = c0 - lado // 2          # primera celda de la región
        for i in range(n):
            for j in range(n):
                dentro = i0 <= i < i0 + lado and i0 <= j < i0 + lado
                centro = (i == c0 and j == c0)
                if k == 3:
                    fc = ORANGE if centro else "white"
                else:
                    fc = LORANGE if dentro else "white"
                ax.add_patch(Rectangle((x0 + j, n - 1 - i), 1, 1,
                                       fc=fc, ec=MUTED, lw=0.3))
        izq, der = x0 + i0, x0 + i0 + lado
        aba, arr = n - (i0 + lado), n - i0
        ax.add_patch(Rectangle((izq, aba), lado, lado, fc="none",
                               ec=ORANGE, lw=1.2, zorder=5))
        cajas.append((izq, der, aba, arr))
        ax.text(x0 + n / 2, n + 0.5, tit, ha="center", va="bottom",
                fontsize=8)
        ax.text(x0 + n / 2, -0.6, f"${lado}\\times{lado}$",
                ha="center", va="top", fontsize=8, color=ORANGE)
    # líneas de dependencia entre una región y la siguiente
    for k in range(3):
        izq0, der0, aba0, arr0 = cajas[k]
        izq1, der1, aba1, arr1 = cajas[k + 1]
        for y0, y1 in ((arr0, arr1), (aba0, aba1)):
            ax.plot([der0, izq1], [y0, y1], color=ORANGE, lw=0.6,
                    ls=(0, (3, 2)), zorder=4)
    ax.set_xlim(-0.3, 4 * n + 3 * sep + 0.3)
    ax.set_ylim(-2.3, n + 2.0)
    guardar(fig, "redes_campo_receptivo.pdf")


# ================================================================
# 4) Subajuste, ajuste adecuado y sobreajuste
# ================================================================
def fig_sobreajuste():
    rng = np.random.default_rng(7)
    n = 12
    # puntos espaciados de forma casi uniforme, con un poco de
    # desorden (evita oscilaciones gigantes del polinomio de grado
    # 11 en los extremos y permite mostrarlo completo)
    x = 0.5 - 0.5 * np.cos(np.pi * (np.arange(n) + 0.5) / n)
    x = np.clip(x + rng.normal(0, 0.012, n), 0.01, 0.99)
    x = np.sort(x)
    f = lambda t: np.sin(2 * np.pi * t)
    y = f(x) + rng.normal(0, 0.32, n)
    xs = np.linspace(0, 1, 300)

    fig = plt.figure(figsize=(6.3, 4.1))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05],
                          hspace=0.55, wspace=0.18)
    titulos = ["Subajuste\n(grado 1)", "Ajuste adecuado\n(grado 3)",
               "Sobreajuste\n(grado 11)"]
    grados = [1, 3, 11]
    for k in range(3):
        ax = fig.add_subplot(gs[0, k])
        coef = np.polyfit(2 * x - 1, y, grados[k])
        ax.plot(xs, f(xs), "--", color=MUTED, lw=0.8)
        ax.plot(xs, np.polyval(coef, 2 * xs - 1), color=BLUE, lw=1.2)
        ax.plot(x, y, "o", color=ORANGE, ms=3.2, mec="white",
                mew=0.4, zorder=4)
        ax.set_ylim(-2.0, 2.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(titulos[k], fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if k == 0:
            ax.set_ylabel("salida")
            ax.set_xlabel("entrada")
            ax.text(-0.02, 1.2, "(a)", transform=ax.transAxes,
                    fontsize=8, fontweight="bold", va="bottom")
    # leyenda de la fila (a)
    from matplotlib.lines import Line2D
    asas = [
        Line2D([], [], marker="o", ls="", color=ORANGE, ms=3.8,
               mec="white", mew=0.4),
        Line2D([], [], ls="--", color=MUTED, lw=0.9),
        Line2D([], [], color=BLUE, lw=1.3),
    ]
    fig.legend(asas, ["datos de entrenamiento (con ruido)",
                      "relación real", "modelo ajustado"],
               loc="center", bbox_to_anchor=(0.5, 0.515), ncol=3,
               frameon=False, fontsize=6.8, handlelength=1.6,
               columnspacing=1.8)

    ax = fig.add_subplot(gs[1, :])
    t = np.arange(0, 101)
    train = 0.75 * np.exp(-t / 14) + 0.05
    val = (0.75 * np.exp(-t / 11) + 0.17
           + 0.0000065 * np.clip(t - 18, 0, None) ** 2.6)
    mejor = int(np.argmin(val))
    ax.plot(t, train, color=BLUE, lw=1.2, label="entrenamiento")
    ax.plot(t, val, color=ORANGE, lw=1.2, label="validación")
    ax.axvline(mejor, color=MUTED, lw=0.7, ls=":")
    ax.plot([mejor], [val[mejor]], "o", color=ORANGE, ms=4,
            mec=INK, mew=0.6, zorder=5)
    ax.annotate("mejor época", (mejor, val[mejor]),
                xytext=(mejor - 3, val[mejor] + 0.38),
                ha="right", fontsize=7,
                arrowprops=dict(arrowstyle="-", lw=0.5,
                                color=MUTED))
    ax.annotate("", xy=(97, 0.52), xytext=(97, 0.1),
                arrowprops=dict(arrowstyle="<->", lw=0.6,
                                color=INK))
    ax.text(95, 0.31, "brecha\ncrece", ha="right", va="center",
            fontsize=7)
    ax.set_xlabel("época de entrenamiento")
    ax.set_ylabel("error")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, loc="upper right", fontsize=7)
    ax.text(0.0, 1.03, "(b)", transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    guardar(fig, "redes_sobreajuste.pdf")


# ================================================================
# 7) Optimizadores sobre un mismo valle (SGD, momentum, Adam)
# ================================================================
def fig_optimizadores():
    c1, c2 = 1.0, 8.0
    L2 = lambda x, y: 0.5 * (c1 * x ** 2 + c2 * y ** 2)
    grad = lambda p: np.array([c1 * p[0], c2 * p[1]])
    p0 = np.array([-3.2, 1.0])

    def sgd(eta, n):
        p, out = p0.copy(), [p0.copy()]
        for _ in range(n):
            p = p - eta * grad(p)
            out.append(p.copy())
        return np.array(out)

    def momentum(eta, beta, n):
        p, v, out = p0.copy(), np.zeros(2), [p0.copy()]
        for _ in range(n):
            v = beta * v + grad(p)
            p = p - eta * v
            out.append(p.copy())
        return np.array(out)

    def adam(eta, n, b1=0.9, b2=0.999, eps=1e-8):
        p, m, v, out = p0.copy(), np.zeros(2), np.zeros(2), [p0.copy()]
        for t in range(1, n + 1):
            g = grad(p)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)
            p = p - eta * mh / (np.sqrt(vh) + eps)
            out.append(p.copy())
        return np.array(out)

    n = 20
    rutas = [
        ("SGD", sgd(0.10, n), MUTED, "o"),
        ("SGD con momentum", momentum(0.03, 0.9, n), BLUE, "s"),
        ("Adam", adam(0.20, n), ORANGE, "^"),
    ]
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    xx, yy = np.meshgrid(np.linspace(-4.6, 4.6, 400),
                         np.linspace(-1.9, 1.9, 400))
    ax.contour(xx, yy, L2(xx, yy),
               levels=[0.2, 0.6, 1.4, 2.8, 4.6, 6.8],
               colors=MUTED, linewidths=0.5, alpha=0.8)
    for nombre, r, col, mk in rutas:
        ax.plot(*r.T, "-", color=col, lw=0.9, zorder=3)
        ax.plot(*r.T, mk, color=col, ms=2.6, zorder=4,
                label=nombre)
    ax.plot(*p0, "s", color=INK, ms=3.5, zorder=5)
    ax.text(p0[0] + 0.1, p0[1] + 0.12, "inicio", fontsize=7)
    ax.plot([0], [0], "*", color=INK, ms=6, zorder=5)
    ax.text(0.2, -0.3, "mínimo", fontsize=7)
    ax.set_xlabel(r"parámetro $w_1$")
    ax.set_ylabel(r"parámetro $w_2$")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-3.95, 3.95)
    ax.set_ylim(-1.5, 1.7)
    ax.legend(loc="lower right", frameon=False, fontsize=7,
              handlelength=1.8, borderaxespad=0.1)
    guardar(fig, "redes_optimizadores.pdf")
    for nombre, r, *_ in rutas:
        print(nombre, "pasos hasta |p|<0.3:",
              next((i for i, q in enumerate(r)
                    if np.hypot(*q) < 0.3), None),
              "final", np.round(r[-1], 3))


# ================================================================
# 8) Búsqueda de hiperparámetros: grilla, aleatoria, guiada
# ================================================================
def fig_hiperparametros():
    # Pérdida de validación sintética sobre dos hiperparámetros
    # (x: tasa de aprendizaje, en escala logarítmica; y: tamaño
    # del lote).
    # Es mucho más sensible a x que a y.
    x0, y0 = 0.62, 0.40
    f = lambda x, y: 1.0 * ((x - x0) / 0.20) ** 2 \
        + 0.35 * ((y - y0) / 0.45) ** 2
    xx, yy = np.meshgrid(np.linspace(0, 1, 300),
                         np.linspace(0, 1, 300))
    zz = f(xx, yy)
    niveles = [0.1, 0.35, 0.8, 1.5, 2.6, 4.0, 6.0]

    rng = np.random.default_rng(7)
    g = np.array([0.17, 0.5, 0.83])
    grilla = np.array([(x, y) for x in g for y in g])
    azar = rng.uniform(0.04, 0.96, size=(9, 2))
    # guiada: 4 ensayos al azar y luego propuestas cerca de los
    # mejores resultados, cada vez más concentradas
    guia = list(rng.uniform(0.04, 0.96, size=(4, 2)))
    sigma = 0.16
    for _ in range(5):
        buenos = sorted(guia, key=lambda q: f(*q))[:2]
        centro = np.mean(buenos, axis=0)
        guia.append(np.clip(centro + rng.normal(0, sigma, 2),
                            0.03, 0.97))
        sigma *= 0.7
    guia = np.array(guia)

    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.35),
                            gridspec_kw={"wspace": 0.12})
    datos = [("(a) Grilla", grilla), ("(b) Aleatoria", azar),
             ("(c) Guiada", guia)]
    for ax, (tit, pts) in zip(axs, datos):
        ax.contourf(xx, yy, zz, levels=[0] + niveles + [99],
                    colors=["#e3edf8", "#ecf2f9", "#f3f6fa",
                            "#f8f9fb", "#fbfbfc", "#fdfdfd",
                            "#ffffff", "#ffffff"], alpha=1)
        ax.contour(xx, yy, zz, levels=niveles, colors=MUTED,
                   linewidths=0.45)
        v = np.array([f(*q) for q in pts])
        ib = int(v.argmin())
        for k, q in enumerate(pts):
            t = k / (len(pts) - 1) if "Guiada" in tit else 1.0
            col = (1 - t) * np.array(mpl.colors.to_rgb("#9dc2ee")) \
                + t * np.array(mpl.colors.to_rgb("#1f5fb0"))
            ax.plot(*q, "o", color=col, ms=3.8, zorder=3)
        ax.plot(*pts[ib], "o", color=ORANGE, ms=5.2, zorder=4,
                mec=INK, mew=0.6)
        ax.plot([x0], [y0], "*", color=INK, ms=5.5, zorder=5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(tit, fontsize=8, loc="left", pad=3,
                     fontweight="bold")
        ax.set_xlabel("tasa de aprendizaje")
    axs[0].set_ylabel("tamaño del lote")
    guardar(fig, "redes_hiperparametros.pdf")
    for nombre, pts in datos:
        print(nombre, "mejor pérdida:",
              round(min(f(*q) for q in pts), 3))


if __name__ == "__main__":
    fig_neurona_mlp()
    fig_neurona()
    fig_mlp()
    fig_activaciones()
    fig_descenso()
    fig_operaciones_cnn()
    fig_campo_receptivo_capas()
    fig_sobreajuste()
    fig_optimizadores()
    fig_hiperparametros()
