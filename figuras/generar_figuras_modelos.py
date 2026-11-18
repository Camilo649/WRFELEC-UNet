# -*- coding: utf-8 -*-
"""
Figuras del Capítulo 4 (Modelos).

Uso (desde la carpeta de la monografía):
    python3 figuras/generar_figuras_modelos.py

Genera en img/:
    comparacion_reconstrucciones.pdf  un mismo instante estimado por
                                    los dos últimos modelos (la
                                    Ext U-Net Comb V2.1 está
                                    en el .npz, pero no se dibuja:
                                    no está optimizada para el
                                    conjunto extendido)

La segunda figura lee figuras/datos/comparacion_instante.npz, que
se genera en la notebook con figuras/celda_comparacion_modelos.py.
Si ese archivo no existe, dibuja una versión provisoria con el
instante de figuras/datos/instante_1924.npz (solo el valor real,
Bob y el modelo Temp) y deja el resto de los paneles marcados como
pendientes.
"""
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# Reutiliza el estilo, los límites provinciales y la paleta de las
# figuras con un instante real.
from generar_figuras_instante import (  # noqa: E402
    limites, coma, guardar, SEQ_RAYOS, INK, MUTED, LIGHT, M_N)

# ---------------------------------------------------------------
# Comparación de reconstrucciones
# ---------------------------------------------------------------
def _panel(ax, M, titulo, vmax, sub=None, color=None):
    if M is None:
        ax.set_facecolor("white")
        ax.set_xlim(-0.5, M_N - 0.5)
        ax.set_ylim(-0.5, M_N - 0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for lado in ax.spines.values():
            lado.set_color("#c9c9c4")
            lado.set_linestyle((0, (3, 2)))
            lado.set_linewidth(0.8)
        ax.text(0.5, 0.5, "pendiente", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color=MUTED,
                style="italic")
    else:
        im = ax.imshow(M, origin="lower", cmap=SEQ_RAYOS, vmin=0,
                       vmax=vmax, interpolation="nearest")
        limites(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        for lado in ax.spines.values():
            lado.set_color(MUTED)
            lado.set_linewidth(0.5)
        ax.im = im
    if sub is not None:
        titulo = titulo + "\n" + sub
    ax.set_title(titulo, fontsize=7.2, color=color or INK,
                 pad=3, linespacing=1.15)


# Ventana de zoom sobre la región con actividad (columna inicial,
# columna final, fila inicial, fila final; origen abajo a la
# izquierda, en la grilla de 297x297).
ZOOM_COMP = (78, 172, 36, 130)


def fig_comparacion():
    ruta = os.path.join(HERE, "datos", "comparacion_instante.npz")
    if os.path.exists(ruta):
        D = np.load(ruta, allow_pickle=True)
        nombres = [str(n) for n in D["nombres"]]
        real = D["y_real"]
        preds = {n: D["preds"][i] for i, n in enumerate(nombres)}
        provisoria = False
    else:
        D = np.load(os.path.join(HERE, "datos", "instante_1924.npz"))
        real = D["y_real"]
        nombres = []
        preds = {"Temp Ext Clip U-Net FACL":
                 np.round(D["pred_real"])}
        provisoria = True

    # Solo se muestra la versión optimizada de cada modelo. En el
    # .npz el modelo optimizado se llama "(Optimized)"; en la
    # monografía no se lo distingue en el nombre. El Temp todavía
    # no es el optimizado: su título va en rojo.
    # La Comb V2.1 ("Ext U-Net Comb V2.1" en el .npz) no se
    # muestra: no se compara con los modelos optimizados.
    orden = [
        "Bob",
        "Ext Clip U-Net FACL (Optimized)",
        "Temp Ext Clip U-Net FACL",
    ]
    etiqueta = {
        "Bob": "Bob",
        "Ext U-Net Comb V2.1":
            "Ext Clip\nU-Net Comb V2.1",
        "Ext Clip U-Net FACL (Optimized)":
            "Ext Clip\nU-Net FACL",
        "Temp Ext Clip U-Net FACL":
            "Temp Ext Clip\nU-Net FACL",
    }
    rojo = "#c00000"
    preds["Bob"] = np.zeros_like(real)
    vmax = float(max([real.max()] + [preds[n].max() for n in orden
                                     if n in preds]))
    vmax = min(vmax, 150.0)

    fig = plt.figure(figsize=(6.8, 2.55))
    gs = fig.add_gridspec(
        2, 4, height_ratios=[1, 0.07], wspace=0.08,
        hspace=0.32, left=0.01, right=0.99, top=0.86,
        bottom=0.16)
    # una fila de 4 paneles: WRF-ELEC, Bob y los dos modelos
    axs = [fig.add_subplot(gs[0, k]) for k in range(4)]
    _panel(axs[0], real, "WRF-ELEC", vmax)
    im = axs[0].im
    for ax, nom in zip(axs[1:], orden):
        _panel(ax, preds.get(nom), etiqueta[nom], vmax,
               color=None)
    c0, c1, f0, f1 = ZOOM_COMP
    for ax in axs:
        ax.set_xlim(c0 - 0.5, c1 - 0.5)
        ax.set_ylim(f0 - 0.5, f1 - 0.5)
    cax = fig.add_subplot(gs[1, 1:3])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.locator = mpl.ticker.MaxNLocator(7)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=7.5, length=2.5, width=0.4,
                      labelcolor=MUTED)
    cb.outline.set_visible(False)
    cb.set_label("LIGHTDENS", fontsize=8, color=MUTED, labelpad=2)
    if provisoria:
        fig.text(0.5, 0.005,
                 "FIGURA PROVISORIA: falta correr "
                 "celda_comparacion_modelos.py",
                 ha="center", fontsize=6.5, color="#b03030")
    guardar(fig, "comparacion_reconstrucciones.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_comparacion()
