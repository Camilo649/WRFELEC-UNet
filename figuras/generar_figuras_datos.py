"""
Figuras del Capítulo 3 (Datos) de la monografía.

Uso (desde la carpeta de la monografía):
    python3 figuras/generar_figuras_datos.py

Genera, en img/:
    imp_corrida1.pdf   importancia por permutación, 6 variables
                       (U-Net Comb V2, corrida del 13/08/26)
    imp_corrida2.pdf   importancia por permutación, 3 variables
                       (Ext Clip U-Net FACL, corrida del 16/10/26)
    imp_final.pdf      importancia por permutación del modelo final
                       (9 canales); se genera solo cuando se completan
                       los valores de IMP_FINAL más abajo.
    correlaciones.pdf  mapas de calor de correlación (Pearson) entre
                       las variables y LIGHTDENS (todos los píxeles
                       y solo los píxeles con descargas)
    distribuciones.pdf distribución acumulada de cada variable con
                       y sin descargas

Las dos últimas leen figuras/datos/analisis_6var.npz, que genera
figuras/celda_analisis_variables.py desde Datasets.ipynb. Si el
archivo no existe, correlaciones.pdf usa los valores de tres
variables copiados de la bitácora (CORR_*) y no se hace
distribuciones.pdf.

Los valores de las dos primeras corridas se copiaron de la
bitácora (media y desvío de 100 permutaciones por variable).
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

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

INK = "#222222"
MUTED = "#8a8a85"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
    OUT = os.path.join(HERE, "..", "img")
except NameError:   # dentro de una notebook no existe __file__
    OUT = "img"     # relativo a donde corre la notebook
os.makedirs(OUT, exist_ok=True)


def guardar(fig, nombre):
    ruta = os.path.join(OUT, nombre)
    fig.savefig(ruta, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(ruta.replace(".pdf", ".png"), bbox_inches="tight",
                pad_inches=0.03, dpi=200)
    plt.close(fig)
    print("OK", ruta)


def coma(x, nd):
    """Formato con coma decimal y signo menos tipográfico."""
    s = f"{x:.{nd}f}".replace(".", ",")
    return s.replace("-", "−")


# ---------------------------------------------------------------
# Resultados (media, desvío) de 100 permutaciones por variable.
# `conservada` indica si la variable pasó al modelo siguiente.
# ---------------------------------------------------------------
IMP_CORRIDA1 = {   # U-Net Comb V2, 13/08/26 (6 variables)
    "HIELO": (22.037173, 2.355857, True),
    "MDBZ": (8.811185, 1.990643, True),
    "CTT": (2.479110, 0.706205, True),
    "LPI": (0.863063, 0.729587, False),
    "WMAX": (0.762580, 0.740887, False),
    "AGUA": (-0.337534, 0.819896, False),
}
E0_CORRIDA1 = 109.280403   # pérdida de validación sin permutar
E0_CORRIDA2 = 0.046355
E0_CORRIDA2_ULT = 0.055700   # pérdida de validación, época 140
E0_FINAL = 0.005871393190472203


def en_porcentaje(datos, e0):
    """Pasa (media, desvío, flag) a porcentaje de la pérdida e0."""
    return {k: (m / e0 * 100, sd / e0 * 100, ok)
            for k, (m, sd, ok) in datos.items()}


IMP_CORRIDA2 = {   # Ext Clip U-Net FACL, 16/10/26 (3 variables)
    "MDBZ": (0.359318, 0.002838, True),
    "HIELO": (0.091167, 0.000464, True),
    "CTT": (0.008494, 0.000128, True),
}
IMP_CORRIDA2_ULT = {   # ídem, con los pesos de la última época (140)
    "MDBZ": (0.329372, 0.002682, True),
    "HIELO": (0.172973, 0.001467, True),
    "CTT": (0.026395, 0.000449, True),
}
# Modelo final (9 canales, ensayo 100, época 47): 100 permutaciones
# por canal, medidas con el MAE normalizado de validación. Los
# valores salen de datos/pfi_oficial_trial0100.npz y se expresan en
# unidades de 1e-3. `True` marca los canales en t; `False`, las
# diferencias.
def cargar_imp_final():
    ruta = os.path.join(HERE, "datos", "pfi_oficial_trial0100.npz") \
        if "HERE" in globals() else "pfi_oficial_trial0100.npz"
    if not os.path.exists(ruta):
        return None
    z = np.load(ruta, allow_pickle=True)
    D = z["deltas"] / float(z["base"]) * 100
    nombres = {
        "MDBZ_t": "MDBZ$_t$", "HIELO_t": "HIELO$_t$",
        "CTT_t": "CTT$_t$",
        "MDBZ_d1": "MDBZ$_{\\Delta1}$", "HIELO_d1": "HIELO$_{\\Delta1}$",
        "CTT_d1": "CTT$_{\\Delta1}$",
        "MDBZ_d2": "MDBZ$_{\\Delta2}$", "HIELO_d2": "HIELO$_{\\Delta2}$",
        "CTT_d2": "CTT$_{\\Delta2}$"}
    return {nombres[str(c)]: (D[:, i].mean(), D[:, i].std(),
                              str(c).endswith("_t"))
            for i, c in enumerate(z["nombres"])}


IMP_FINAL = cargar_imp_final()


def grafico_importancia(datos, nombre, nd, xmax, leyenda=False,
                        ancho=3.5, alto=None, xlabel=None,
                        etiquetas=("Variable conservada",
                                   "Variable descartada"),
                        e0=None, color_no=MUTED):
    items = sorted(datos.items(), key=lambda kv: -kv[1][0])
    n = len(items)
    alto = alto or (0.55 + 0.30 * n)
    fig, ax = plt.subplots(figsize=(ancho, alto))
    ys = list(range(n))[::-1]
    for y, (var, (m, s, ok)) in zip(ys, items):
        ax.barh(y, m, height=0.58, color=BLUE if ok else color_no,
                zorder=2)
        ax.errorbar(m, y, xerr=s, color=INK, lw=0.7, capsize=2.2,
                    capthick=0.7, zorder=3)
        x_txt = max(m + s, 0) + 0.018 * xmax
        ax.text(x_txt, y, f"{coma(m, nd)} ± {coma(s, nd)}",
                va="center", ha="left", color=INK, fontsize=8.2)
    ax.axvline(0, color=INK, lw=0.6, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([v for v, _ in items], color=INK)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2.5, width=0.6, labelcolor=MUTED)
    xmin = min(0, min(m - s for m, s, _ in datos.values()) * 1.15)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.6, n - 0.4)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_xlabel(xlabel or
                  r"Aumento de la pérdida de validación (%)",
                  color=INK)
    if e0 is not None:
        ax.set_title(e0, loc="right", fontsize=8.2, color=INK, pad=3)
    if leyenda:
        ax.legend(handles=[
            Patch(color=BLUE, label=etiquetas[0]),
            Patch(color=color_no, label=etiquetas[1])],
            loc="lower right", frameon=False, fontsize=8.2,
            handlelength=1.0, borderaxespad=0.2)
    # separador de miles con punto en los ticks
    ax.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(
            lambda v, _: coma(v, 0 if xmax > 5 else 1)
            if xmax > 5 else coma(v, 2)))
    guardar(fig, nombre)


# ---------------------------------------------------------------
# Correlaciones (valores leídos de los mapas de calor de la
# bitácora; orden: MDBZ, HIELO, CTT, LIGHTDENS)
# ---------------------------------------------------------------
CORR_VARS = ["MDBZ", "HIELO", "CTT", "LIGHTDENS"]
CORR_TODOS = np.array([
    [1.00, 0.69, -0.60, 0.35],
    [0.69, 1.00, -0.45, 0.46],
    [-0.60, -0.45, 1.00, -0.18],
    [0.35, 0.46, -0.18, 1.00]])
CORR_CON_RAYOS = np.array([
    [1.00, 0.56, -0.16, 0.31],
    [0.56, 1.00, -0.31, 0.29],
    [-0.16, -0.31, 1.00, -0.16],
    [0.31, 0.29, -0.16, 1.00]])


def cargar_analisis():
    ruta = os.path.join(HERE, "datos", "analisis_6var.npz") \
        if "HERE" in globals() else "analisis_6var.npz"
    if not os.path.exists(ruta):
        return None
    return np.load(ruta, allow_pickle=True)


def fig_correlaciones(A=None):
    if A is None:
        nombres, M1, M2 = CORR_VARS, CORR_TODOS, CORR_CON_RAYOS
    else:
        nombres = [str(v) for v in A["vars"]]
        M1, M2 = A["corr_todos"], A["corr_con"]
    n = len(nombres)
    cmap = LinearSegmentedColormap.from_list(
        "div", [BLUE, "#f4f4f2", ORANGE])
    lado = 6.4 if n <= 4 else 6.6
    fig, axs = plt.subplots(1, 2, figsize=(lado, 0.5 * lado
                            if n <= 4 else 0.52 * lado),
                            gridspec_kw={"wspace": 0.06})
    titulos = ["(a) Todas las columnas",
               "(b) Solo columnas con descargas"]
    fs = 8.2 if n <= 4 else 6.6
    for ax, M, tit in zip(axs, [M1, M2], titulos):
        im = ax.imshow(M, cmap=cmap, vmin=-1, vmax=1)
        for i in range(n):
            for j in range(n):
                v = M[i, j]
                ax.text(j, i, coma(v, 2), ha="center", va="center",
                        fontsize=fs,
                        color="white" if abs(v) > 0.75 else INK)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(nombres, fontsize=7, color=INK,
                           rotation=0 if n <= 4 else 45,
                           ha="center" if n <= 4 else "right")
        ax.set_yticklabels(nombres if ax is axs[0] else [],
                           fontsize=7, color=INK)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.4)
        ax.tick_params(which="minor", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(tit, fontsize=8.6, color=INK, pad=5)
    cb = fig.colorbar(im, ax=axs, shrink=0.78, pad=0.03,
                      ticks=[-1, -0.5, 0, 0.5, 1])
    cb.ax.set_yticklabels([coma(t, 1) for t in
                           [-1, -0.5, 0, 0.5, 1]], fontsize=7.4)
    cb.set_label("Coeficiente de correlación", fontsize=8,
                 color=INK)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, width=0.5)
    guardar(fig, "correlaciones.pdf")


UNIDADES = {"MDBZ": "dBZ", "HIELO": "kg/kg", "AGUA": "kg/kg",
            "CTT": "°C", "LPI": "J/kg", "WMAX": "m/s"}


def fig_distribuciones(A):
    """Distribución acumulada de cada variable, con y sin descargas.

    Se usa la acumulada porque no depende del ancho de los
    intervalos (los de analisis_6var.npz no son uniformes)."""
    nombres = [str(v) for v in A["vars"]][:-1]
    n = len(nombres)
    cols = 3
    filas = int(np.ceil(n / cols))
    fig, axs = plt.subplots(filas, cols, figsize=(6.6, 1.75 * filas),
                            squeeze=False,
                            gridspec_kw={"hspace": 0.62,
                                         "wspace": 0.28})
    for k, nom in enumerate(nombres):
        ax = axs[k // cols][k % cols]
        e = A["bins"][k]
        hc, hs = A["h_con"][k], A["h_sin"][k]
        # el eje se corta en el cuantil 99,9 % de todos los píxeles
        acum = np.cumsum(hc + hs) / (hc + hs).sum()
        xmax = e[1:][np.searchsorted(acum, 0.999)]
        for h, col, lab in ((hs, MUTED, "sin descargas"),
                            (hc, ORANGE, "con descargas")):
            F = np.concatenate([[0], np.cumsum(h) / h.sum()])
            ax.plot(e, F, color=col, lw=1.3, label=lab,
                    drawstyle="steps-post")
        ax.set_xlim(e[0], xmax)
        ax.set_ylim(0, 1.02)
        ax.set_title(nom, fontsize=8.6, color=INK, pad=3)
        ax.set_xlabel(UNIDADES.get(nom, ""), fontsize=7.4,
                      color=INK, labelpad=1)
        ax.tick_params(labelsize=6.8, length=2, width=0.5)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0", "0,5", "1"])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if k % cols == 0:
            ax.set_ylabel("Fracción acumulada", fontsize=7.4,
                          color=INK)
    for k in range(n, filas * cols):
        axs[k // cols][k % cols].axis("off")
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.04))
    guardar(fig, "distribuciones.pdf")


if __name__ == "__main__":
    grafico_importancia(
        en_porcentaje(IMP_CORRIDA1, E0_CORRIDA1), "imp_corrida1.pdf",
        1, 32, leyenda=True, e0=r"$E_0=109{,}28$")
    grafico_importancia(
        en_porcentaje(IMP_CORRIDA2, E0_CORRIDA2), "imp_corrida2.pdf",
        1, 1450, e0=r"$E_0=0{,}0464$")
    grafico_importancia(
        en_porcentaje(IMP_CORRIDA2_ULT, E0_CORRIDA2_ULT),
        "imp_corrida2_ultima.pdf", 1, 1450,
        e0=r"$E_0=0{,}0557$")
    A = cargar_analisis()
    fig_correlaciones(A)
    if A is not None:
        fig_distribuciones(A)
    if IMP_FINAL:
        grafico_importancia(
            IMP_FINAL, "imp_final.pdf", 2,
            max(m + s for m, s, _ in IMP_FINAL.values()) * 1.45,
            leyenda=True,
            xlabel=r"Aumento del MAE de validación (%)",
            etiquetas=("Variable en $t$",
                       r"Diferencia ($\Delta_1$ o $\Delta_2$)"),
            e0=r"$E_0=0{,}00587$", color_no=ORANGE)
