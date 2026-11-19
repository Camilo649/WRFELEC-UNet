# -*- coding: utf-8 -*-
"""
Figuras con un instante real (Figs. 3.1 y 3.4 de la monografía).

Uso (desde la carpeta de la monografía):
    python3 figuras/generar_figuras_instante.py

Lee figuras/datos/instante_1924.npz (04/03/2019, 12:28 UTC, tormenta
de validación) y genera en img/:
    ejemplo_variables.pdf  las seis variables y LIGHTDENS
    deltas_hielo.pdf       HIELO en t, Delta1 y Delta2
    ejemplo_problema.pdf   HIELO y LIGHTDENS simulada
En todos los mapas se superponen los límites provinciales
(datos/limites_provincias.npz, generado con preparar_limites.py).
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

M_N = 297

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif",
                   "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.linewidth": 0.5,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
})
INK, MUTED = "#222222", "#8a8a85"
BLUE, DBLUE, ORANGE, DORANGE = ("#2a78d6", "#0e3a73", "#eb6834",
                                "#7a2a0a")
LIGHT = "#f4f4f2"

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = "figuras"
OUT = os.path.join(HERE, "..", "img")
os.makedirs(OUT, exist_ok=True)
D = np.load(os.path.join(HERE, "datos", "instante_1924.npz"))
X = D["crudo_X"]
NOM = [str(n) for n in D["nombres_canales"]]
canal = lambda n: X[NOM.index(n)]

SEQ = LinearSegmentedColormap.from_list("seq", [LIGHT, BLUE, DBLUE])
SEQ_FRIO = LinearSegmentedColormap.from_list(
    "frio", [DBLUE, BLUE, LIGHT])
SEQ_RAYOS = LinearSegmentedColormap.from_list(
    "rayos", [LIGHT, ORANGE, DORANGE])
DIV = LinearSegmentedColormap.from_list("div", [BLUE, LIGHT, ORANGE])


LIM = np.load(os.path.join(HERE, "datos", "limites_provincias.npz"))
COLOR_LIM = "#6f6f6a"


def limites(ax):
    """Superpone los límites provinciales (en píxeles de la grilla)."""
    xl, yl = ax.get_xlim(), ax.get_ylim()
    ax.plot(LIM["x"], LIM["y"], color=COLOR_LIM, lw=0.45,
            solid_joinstyle="round", zorder=2)
    ax.set_xlim(xl)
    ax.set_ylim(yl)


def coma(x, nd=0):
    s = f"{x:.{nd}f}".replace(".", ",")
    return s.replace("-", "−")


def guardar(fig, nombre):
    ruta = os.path.join(OUT, nombre)
    fig.savefig(ruta, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(ruta.replace(".pdf", ".png"), bbox_inches="tight",
                pad_inches=0.03, dpi=200)
    plt.close(fig)
    print("OK", ruta)


def panel(ax, M, cmap, titulo, vmin=None, vmax=None, nd=0):
    im = ax.imshow(M, origin="lower", cmap=cmap, vmin=vmin,
                   vmax=vmax, interpolation="nearest")
    limites(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    for lado in ax.spines.values():
        lado.set_color(MUTED)
        lado.set_linewidth(0.5)
    ax.set_title(titulo, fontsize=8, color=INK, pad=3)
    cb = ax.figure.colorbar(im, ax=ax, orientation="horizontal",
                            fraction=0.05, pad=0.04, aspect=18,
                            format=mpl.ticker.FuncFormatter(
                                lambda v, _: coma(v, nd)))
    cb.locator = mpl.ticker.MaxNLocator(3)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=6.5, length=2, width=0.4,
                      labelcolor=MUTED)
    cb.outline.set_visible(False)
    return cb


def _titulo(ax, letra, nombre):
    ax.set_title(r"$\mathbf{(%s)}$ %s" % (letra, nombre), loc="left",
                 fontsize=8.5, color=INK, pad=4)


def _marco(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(LIGHT)
    ax.set_xlim(-0.5, M_N - 0.5)
    ax.set_ylim(-0.5, M_N - 0.5)
    ax.set_aspect("equal")
    for lado in ax.spines.values():
        lado.set_color("#c9c9c4")
        lado.set_linewidth(0.6)


def _barra(fig, ax, im, unidad, nd):
    cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                      fraction=0.045, pad=0.05, aspect=22,
                      format=mpl.ticker.FuncFormatter(
                          lambda v, _: coma(v, nd)))
    cb.locator = mpl.ticker.MaxNLocator(3)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=6.5, length=2, width=0.4,
                      labelcolor=MUTED, color="#c9c9c4")
    cb.outline.set_visible(False)
    cb.set_label(unidad, fontsize=6.8, color=MUTED, labelpad=2)


def fig_variables():
    """Fig. 3.1: cuatro paneles arriba y tres abajo, centrados. Las
    variables de entrada se muestran sin marcas encima."""
    global M_N
    y = D["y_real"]
    M_N = y.shape[0]
    fig = plt.figure(figsize=(6.6, 4.6))
    gs = fig.add_gridspec(2, 8, hspace=0.30, wspace=0.55,
                          left=0.01, right=0.99, top=0.95, bottom=0.04)
    celdas = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6], gs[0, 6:8],
              gs[1, 1:3], gs[1, 3:5], gs[1, 5:7]]
    datos = [
        ("a", "MDBZ", "dBZ", canal("MDBZ_t"), SEQ, 0),
        ("b", "HIELO", "kg/kg", canal("HIELO_t"), SEQ, 3),
        ("c", "AGUA", "kg/kg", D["extra_AGUA"], SEQ, 4),
        ("d", "CTT", "°C", canal("CTT_t"), SEQ_FRIO, 0),
        ("e", "LPI", "J/kg", D["extra_LPI"], SEQ, 0),
        ("f", "WMAX", "m/s", D["extra_WMAX"], SEQ, 0),
    ]
    fi, co = np.nonzero(y > 0)
    for celda, (let, nom, uni, M, cm, nd) in zip(celdas, datos):
        ax = fig.add_subplot(celda)
        im = ax.imshow(M, origin="lower", cmap=cm,
                       interpolation="nearest")
        _marco(ax)
        limites(ax)
        _titulo(ax, let, nom)
        _barra(fig, ax, im, uni, nd)
    # (g) LIGHTDENS: un punto por columna con descargas, cuyo tamaño
    # y color crecen con el valor
    ax = fig.add_subplot(celdas[6])
    _marco(ax)
    limites(ax)
    v = y[fi, co]
    sc = ax.scatter(co, fi, c=v, cmap=SEQ_RAYOS, vmin=0, vmax=v.max(),
                    s=1.5 + 9 * (v / v.max()) ** 0.7,
                    edgecolors="none", zorder=3)
    _titulo(ax, "g", "LIGHTDENS")
    _barra(fig, ax, sc, "descargas por columna", 0)
    guardar(fig, "ejemplo_variables.pdf")


ZOOM_DELTAS = (100, 230, 35, 165)


def fig_deltas():
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.4),
                            gridspec_kw={"wspace": 0.12})
    h, d1, d2 = (canal("HIELO_t"), canal("HIELO_d1"),
                 canal("HIELO_d2"))
    # Zoom sobre la región con más hielo (columnas y filas de la
    # grilla de 297x297, con el origen abajo a la izquierda).
    c0, c1, f0, f1 = ZOOM_DELTAS
    sl = (slice(f0, f1), slice(c0, c1))
    r = max(np.abs(d1[sl]).max(), np.abs(d2[sl]).max())
    panel(axs[0], h, SEQ, "(a) HIELO en $t$ (kg/kg)", 0, h[sl].max(),
          nd=3)
    panel(axs[1], d1, DIV, "(b) $\\Delta_1$ (kg/kg)", -r, r, nd=3)
    panel(axs[2], d2, DIV, "(c) $\\Delta_2$ (kg/kg)", -r, r, nd=3)
    for ax in axs:
        ax.set_xlim(c0 - 0.5, c1 - 0.5)
        ax.set_ylim(f0 - 0.5, f1 - 0.5)
    guardar(fig, "deltas_hielo.pdf")


def fig_problema():
    fig, axs = plt.subplots(1, 2, figsize=(4.5, 2.5),
                            gridspec_kw={"wspace": 0.12})
    y = D["y_real"]
    cb = panel(axs[0], canal("HIELO_t"), SEQ,
               "(a) Razón de mezcla del hielo", nd=3)
    cb.set_label("kg/kg", fontsize=6.8, color=MUTED, labelpad=2)
    cb = panel(axs[1], y, SEQ_RAYOS, "(b) Densidad de descargas",
               0, y.max())
    cb.set_label("descargas por columna", fontsize=6.8, color=MUTED, labelpad=2)
    guardar(fig, "ejemplo_problema.pdf")


def fig_limitacion_cnn():
    """Limitación de la CNN base: la zona con descargas se extiende
    más allá de lo que alcanza a ver una ventana de 7x7. Se usa una
    célula aislada del instante real: en un píxel del borde de la
    zona con descargas, el núcleo de HIELO queda fuera de la
    ventana."""
    import matplotlib.patheffects as pe
    h = canal("HIELO_t")
    y = D["y_real"]
    k = 3                                    # radio de la ventana 7x7
    sub = h[94:112, 158:178]
    ci, cj = np.unravel_index(sub.argmax(), sub.shape)
    nucleo = (94 + ci, 158 + cj)             # (fila, columna)
    P = (100, 163)                           # píxel del borde de la zona
    dist = float(np.hypot(P[0] - nucleo[0], P[1] - nucleo[1]))
    r0, r1, c0, c1 = 82, 122, 146, 186       # recorte de (a) y (b)
    vmax_h = float(h[r0:r1, c0:c1].max())
    vmax_y = float(y[r0:r1, c0:c1].max())
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]

    fig = plt.figure(figsize=(6.6, 2.75))
    gs = fig.add_gridspec(1, 3, wspace=0.12, left=0.01, right=0.99,
                          top=0.88, bottom=0.17)
    ax_h = fig.add_subplot(gs[0, 0])
    ax_y = fig.add_subplot(gs[0, 1])
    ax_w = fig.add_subplot(gs[0, 2])

    def recorte(ax, M, cmap, vmax):
        im = ax.imshow(M, origin="lower", cmap=cmap, vmin=0,
                       vmax=vmax, interpolation="nearest")
        limites(ax)
        ax.set_xlim(c0 - 0.5, c1 - 0.5)
        ax.set_ylim(r0 - 0.5, r1 - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for lado in ax.spines.values():
            lado.set_color(MUTED)
            lado.set_linewidth(0.5)
        return im

    def titulo(ax, letra, texto):
        ax.set_title(r"$\mathbf{(%s)}$ %s" % (letra, texto),
                     loc="left", fontsize=8.5, color=INK, pad=4)

    im_h = recorte(ax_h, h, SEQ, vmax_h)
    im_y = recorte(ax_y, y, SEQ_RAYOS, vmax_y)
    titulo(ax_h, "a", "HIELO (kg/kg)")
    titulo(ax_y, "b", "LIGHTDENS")

    for ax in (ax_h, ax_y):
        # otras ventanas de 7x7 (en gris), alineadas con la marcada
        x_ini = P[1] - k - 0.5 - 7 * np.ceil((P[1] - k - 0.5 - c0 + 0.5) / 7)
        y_ini = P[0] - k - 0.5 - 7 * np.ceil((P[0] - k - 0.5 - r0 + 0.5) / 7)
        for xx in np.arange(x_ini, c1, 7):
            ax.plot([xx, xx], [r0 - 0.5, r1 - 0.5], color="#7d7d78",
                    lw=0.5, alpha=0.55, zorder=3)
        for yy in np.arange(y_ini, r1, 7):
            ax.plot([c0 - 0.5, c1 - 0.5], [yy, yy], color="#7d7d78",
                    lw=0.5, alpha=0.55, zorder=3)
        for ec, lw, z in (("white", 2.0, 4), (INK, 0.9, 5)):
            ax.add_patch(plt.Rectangle(
                (P[1] - k - 0.5, P[0] - k - 0.5), 7, 7, fill=False,
                ec=ec, lw=lw, zorder=z))
        ax.plot([P[1], nucleo[1]], [P[0], nucleo[0]], color=INK,
                lw=0.8, ls=(0, (2.5, 1.5)), zorder=5,
                path_effects=halo)
        ax.plot(*nucleo[::-1], marker="*", ms=8, mfc="white",
                mec=INK, mew=0.7, zorder=7)
        ax.plot(*P[::-1], marker="o", ms=2.6, mfc=INK, mec="white",
                mew=0.5, zorder=7)

    # valor de LIGHTDENS en el pixel marcado, anotado sobre (b)
    ax_y.annotate("%d" % y[P], xy=P[::-1], xytext=(-26, -22),
                  textcoords="offset points", fontsize=7.5,
                  color=INK, ha="center", va="center", zorder=9,
                  bbox=dict(boxstyle="round,pad=0.25", fc="white",
                            ec=INK, lw=0.5),
                  arrowprops=dict(arrowstyle="-", color=INK, lw=0.6,
                                  shrinkA=0, shrinkB=2))

    for ax, im, nd in ((ax_h, im_h, 3), (ax_y, im_y, 0)):
        cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                          fraction=0.05, pad=0.04, aspect=22,
                          format=mpl.ticker.FuncFormatter(
                              lambda v, _, nd=nd: coma(v, nd)))
        cb.locator = mpl.ticker.MaxNLocator(3)
        cb.update_ticks()
        cb.ax.tick_params(labelsize=6.5, length=2, width=0.4,
                          labelcolor=MUTED)
        cb.outline.set_visible(False)

    # (c) zona de 15x15 alrededor de P: lo que la red ve y lo que no
    m = 7
    zi0, zi1 = P[0] - m, P[0] + m
    zj0, zj1 = P[1] - m, P[1] + m
    ax_w.imshow(h[zi0:zi1 + 1, zj0:zj1 + 1], origin="lower", cmap=SEQ,
                vmin=0, vmax=vmax_h, interpolation="nearest",
                extent=(zj0 - 0.5, zj1 + 0.5, zi0 - 0.5, zi1 + 0.5))
    velo = np.zeros((2 * m + 1, 2 * m + 1, 4))
    velo[..., :3] = (0.35, 0.35, 0.35)
    velo[..., 3] = 0.72
    velo[m - k:m + k + 1, m - k:m + k + 1, 3] = 0.0
    ax_w.imshow(velo, origin="lower", interpolation="nearest",
                extent=(zj0 - 0.5, zj1 + 0.5, zi0 - 0.5, zi1 + 0.5),
                zorder=3)
    for t in np.arange(zj0 - 0.5, zj1 + 1, 1.0):
        ax_w.axvline(t, color="white", lw=0.25, alpha=0.7, zorder=4)
    for t in np.arange(zi0 - 0.5, zi1 + 1, 1.0):
        ax_w.axhline(t, color="white", lw=0.25, alpha=0.7, zorder=4)
    ax_w.add_patch(plt.Rectangle((P[1] - k - 0.5, P[0] - k - 0.5), 7,
                                 7, fill=False, ec=INK, lw=1.1,
                                 zorder=6))
    ax_w.plot(*nucleo[::-1], marker="*", ms=9, mfc="white", mec=INK,
              mew=0.7, zorder=7)
    ax_w.plot(*P[::-1], marker="o", ms=3.2, mfc=INK, mec="white",
              mew=0.5, zorder=7)
    ax_w.set_xlim(zj0 - 0.5, zj1 + 0.5)
    ax_w.set_ylim(zi0 - 0.5, zi1 + 0.5)
    ax_w.set_xticks([])
    ax_w.set_yticks([])
    for lado in ax_w.spines.values():
        lado.set_color(MUTED)
        lado.set_linewidth(0.5)
    titulo(ax_w, "c", "Entrada (HIELO) que alcanza la red")
    guardar(fig, "limitacion_cnn_base.pdf")
    print("núcleo", nucleo, "píxel", P, "distancia (px)",
          round(dist, 2), "LIGHTDENS", float(y[P]),
          "HIELO máx en la ventana", round(float(
              h[P[0]-k:P[0]+k+1, P[1]-k:P[1]+k+1].max()), 4),
          "HIELO del núcleo", round(float(h[nucleo]), 4))


def fig_campo_receptivo():
    """Campo receptivo (a escala) de la CNN base (7x7) y de la U-Net
    del mejor modelo, la Ext Clip U-Net Comb V2.1 (132x132, calculado
    con las dependencias exactas de la arquitectura) para el mismo
    píxel de salida, sobre el dominio."""
    import matplotlib.patheffects as pe
    h = canal("HIELO_t")
    y = D["y_real"]
    sub = h[94:112, 158:178]
    ci, cj = np.unravel_index(sub.argmax(), sub.shape)
    nucleo = (94 + ci, 158 + cj)
    P = (100, 163)
    RF_UNET = 132
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]
    fig = plt.figure(figsize=(6.6, 3.15))
    gs = fig.add_gridspec(1, 2, wspace=0.10, left=0.01, right=0.99,
                          top=0.89, bottom=0.17)
    for n, (M, cmap, tit, nd) in enumerate(
            ((h, SEQ, "HIELO (kg/kg)", 3),
             (y, SEQ_RAYOS, "LIGHTDENS", 0))):
        ax = fig.add_subplot(gs[0, n])
        im = ax.imshow(M, origin="lower", cmap=cmap, vmin=0,
                       vmax=float(M.max()), interpolation="nearest")
        limites(ax)
        ax.set_xlim(-0.5, M.shape[1] - 0.5)
        ax.set_ylim(-0.5, M.shape[0] - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for lado in ax.spines.values():
            lado.set_color(MUTED)
            lado.set_linewidth(0.5)
        ax.set_title(r"$\mathbf{(%s)}$ %s" % ("ab"[n], tit), loc="left",
                     fontsize=8.5, color=INK, pad=4)
        # U-Net del mejor modelo
        a = RF_UNET / 2
        for ec, lw, z in (("white", 2.2, 4), (DORANGE, 1.1, 5)):
            ax.add_patch(plt.Rectangle((P[1] - a - 0.5, P[0] - a - 0.5),
                                       RF_UNET, RF_UNET, fill=False,
                                       ec=ec, lw=lw, zorder=z))
        # CNN base (7x7)
        for ec, lw, z in (("white", 2.2, 6), (INK, 1.0, 7)):
            ax.add_patch(plt.Rectangle((P[1] - 3.5, P[0] - 3.5), 7, 7,
                                       fill=False, ec=ec, lw=lw,
                                       zorder=z))
        ax.plot(*nucleo[::-1], marker="*", ms=7, mfc="white", mec=INK,
                mew=0.6, zorder=8)
        if n == 0:
            ax.annotate("CNN base\n$7\\times7$", xy=(P[1] - 3.5, P[0]),
                        xytext=(P[1] - a - 24, P[0] + 6), fontsize=7,
                        color=INK, ha="center", va="center", zorder=9,
                        arrowprops=dict(arrowstyle="-", lw=0.6,
                                        color=INK),
                        path_effects=halo)
            ax.text(P[1] - a, P[0] + a + 3,
                    "Mejor modelo, $132\\times132$", fontsize=7,
                    color=DORANGE, ha="left", va="bottom", zorder=9,
                    path_effects=halo)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                          fraction=0.05, pad=0.04, aspect=26,
                          format=mpl.ticker.FuncFormatter(
                              lambda v, _, nd=nd: coma(v, nd)))
        cb.locator = mpl.ticker.MaxNLocator(3)
        cb.update_ticks()
        cb.ax.tick_params(labelsize=6.5, length=2, width=0.4,
                          labelcolor=MUTED)
        cb.outline.set_visible(False)
    guardar(fig, "campo_receptivo.pdf")


if __name__ == "__main__":
    fig_limitacion_cnn()
    fig_campo_receptivo()
    fig_problema()
    fig_variables()
    fig_deltas()
