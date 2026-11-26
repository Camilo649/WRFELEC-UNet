"""
Figuras de la sección 2.4 (Función de Pérdida) de la monografía.
Son esquemas didácticos con datos sintéticos: no usan datos del
trabajo.

Uso (desde la carpeta de la monografía):
    python3 figuras/generar_figuras_facl.py

Genera, en img/:
    facl_espectros.pdf    (objetivo, copia desplazada y copia
                           difuminada, con sus espectros de amplitud
                           y los valores de MSE, FAL y FCL)
    facl_cronograma.pdf   (cronograma P(t) y sorteo de FAL/FCL)
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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
MUTED = "#8a8a85"
BLUE = "#2a78d6"      # FCL
ORANGE = "#eb6834"    # FAL
LGRAY = "#ececea"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "img")
os.makedirs(OUT, exist_ok=True)

CMAP = LinearSegmentedColormap.from_list(
    "azul", ["#ffffff", "#d6e4f5", "#2a78d6", "#0b2a5b"])


def guardar(fig, nombre):
    ruta = os.path.join(OUT, nombre)
    fig.savefig(ruta, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(ruta.replace(".pdf", ".png"), bbox_inches="tight",
                pad_inches=0.03, dpi=200)
    plt.close(fig)
    print("OK", ruta)


# ---------------------------------------------------------------
# Las tres pérdidas, tal como se definen en el texto
# ---------------------------------------------------------------
def mse(x, y):
    return float(np.mean((x - y) ** 2))


def fal(x, y):
    F = np.fft.fft2(x, norm="ortho")
    G = np.fft.fft2(y, norm="ortho")
    return float(np.mean((np.abs(F) - np.abs(G)) ** 2))


def fcl(x, y):
    F = np.fft.fft2(x, norm="ortho")
    G = np.fft.fft2(y, norm="ortho")
    num = np.sum(np.real(F * np.conj(G)))
    den = np.sqrt(np.sum(np.abs(F) ** 2) * np.sum(np.abs(G) ** 2))
    return float(1.0 - num / den)


def campo_sintetico(n=64):
    """Tres 'núcleos' gaussianos de distinta intensidad sobre cero."""
    yy, xx = np.mgrid[0:n, 0:n]
    nucleos = [(20, 22, 1.7, 1.0), (40, 38, 1.4, 0.7),
               (30, 50, 1.2, 0.45)]
    x = np.zeros((n, n))
    for cy, cx, s, a in nucleos:
        x += a * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2)
                        / (2 * s ** 2))
    return x


def gauss_fft(x, sigma):
    """Difuminado gaussiano (circular) mediante la FFT."""
    n = x.shape[0]
    f = np.fft.fftfreq(n)
    k2 = f[:, None] ** 2 + f[None, :] ** 2
    h = np.exp(-2 * (np.pi * sigma) ** 2 * k2)
    return np.real(np.fft.ifft2(np.fft.fft2(x) * h))


# ================================================================
# 1) Qué ve cada término: desplazamiento y difuminado
# ================================================================
def fig_espectros():
    x = campo_sintetico()
    xs = np.roll(x, (4, 5), axis=(0, 1))      # mismo campo, corrido
    xb = gauss_fft(x, 3.0)                    # campo difuminado
    imgs = [x, xs, xb]
    titulos = ["(a) Objetivo", "(b) Copia desplazada",
               "(c) Copia difuminada"]

    def amp(z):
        return np.fft.fftshift(np.abs(np.fft.fft2(z, norm="ortho")))
    lg = [np.log10(amp(z) + 1e-3) for z in imgs]
    vmin, vmax = -3.0, max(l.max() for l in lg)

    fig, ax = plt.subplots(
        2, 3, figsize=(6.3, 4.05),
        gridspec_kw={"hspace": 0.10, "wspace": 0.08})
    for j in range(3):
        a = ax[0, j]
        im_a = a.imshow(imgs[j], cmap=CMAP, vmin=0, vmax=1,
                        interpolation="nearest")
        a.set_title(titulos[j], fontsize=8, color=INK, pad=3)
        a.set_xticks([])
        a.set_yticks([])
        b = ax[1, j]
        im_b = b.imshow(lg[j], cmap=CMAP, vmin=vmin, vmax=vmax,
                        interpolation="nearest")
        b.set_xticks([])
        b.set_yticks([])
        for s in list(a.spines.values()) + list(b.spines.values()):
            s.set_color(MUTED)
            s.set_linewidth(0.5)
    # una barra de color por fila (cada fila tiene su escala)
    for fila, im, rot in ((0, im_a, "Intensidad"),
                          (1, im_b, r"$\log_{10}|F|$")):
        cb = fig.colorbar(im, ax=ax[fila, :].tolist(), shrink=0.9,
                          aspect=18, pad=0.02, fraction=0.03)
        cb.set_label(rot, fontsize=7.5, color=INK)
        cb.ax.tick_params(labelsize=7, width=0.5, length=2)
        cb.outline.set_linewidth(0.5)
        cb.outline.set_edgecolor(MUTED)
    ax[0, 0].set_ylabel("Imagen", fontsize=8, color=INK)
    ax[1, 0].set_ylabel("Espectro de amplitud\n$|F|$ (escala log)",
                        fontsize=8, color=INK)

    # valores de cada pérdida respecto del objetivo
    textos = ["(referencia)"]
    def f4(v):
        return "0" if v < 5e-5 else "{:.4f}".format(v)
    for z in (xs, xb):
        textos.append(
            "MSE = {}\nFAL = {}\nFCL = {:.3f}".format(
                f4(mse(x, z)), f4(fal(x, z)), fcl(x, z)))
    for j in range(3):
        ax[1, j].text(0.5, -0.07, textos[j], transform=ax[1, j]
                      .transAxes, ha="center", va="top",
                      fontsize=7.5, color=INK, linespacing=1.35)
    guardar(fig, "facl_espectros.pdf")
    return {"mse_s": mse(x, xs), "fal_s": fal(x, xs),
            "fcl_s": fcl(x, xs), "mse_b": mse(x, xb),
            "fal_b": fal(x, xb), "fcl_b": fcl(x, xb),
            "pico_b": xb.max()}


# ================================================================
# 2) Cronograma P(t) y sorteo de la pérdida en cada paso
# ================================================================
def fig_cronograma(alpha=0.2, T=100, seed=3):
    t = np.arange(T + 1)
    t_fin = (1 - alpha) * T
    P = np.clip(1 - t / t_fin, 0, 1)

    rng = np.random.default_rng(seed)
    pasos = np.arange(T)
    P_paso = np.clip(1 - pasos / t_fin, 0, 1)
    p = rng.random(T)
    es_fal = p > P_paso

    fig, (a, b) = plt.subplots(
        2, 1, figsize=(6.3, 3.2),
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.55})
    # (a) P(t)
    a.fill_between(t, P, 1, color=ORANGE, alpha=0.18, lw=0)
    a.fill_between(t, 0, P, color=BLUE, alpha=0.18, lw=0)
    a.plot(t, P, color=INK, lw=1.2)
    a.axvline(t_fin, ymax=0.33, color=MUTED, lw=0.7, ls=":")
    a.set_xlim(0, T)
    a.set_ylim(0, 1.02)
    a.set_xticks([0, t_fin, T])
    a.set_xticklabels(["0", r"$(1-\alpha)\,T$", r"$T$"])
    a.set_yticks([0, 0.5, 1])
    a.set_xlabel("Paso de entrenamiento $t$", fontsize=8)
    a.set_ylabel("$P(t)$", fontsize=8)
    a.text(8, 0.12, "Probabilidad de elegir FCL "
           "$=P(t)$", color=INK, fontsize=7.5, ha="left")
    a.text(32, 0.9, "Probabilidad de elegir FAL "
           "$=1-P(t)$", color=INK, fontsize=7.5, ha="left",
           va="center")
    a.annotate("", xy=(T, 0.07), xytext=(t_fin, 0.07),
               arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7))
    a.text((t_fin + T) / 2, 0.14, r"$\alpha T$ pasos: solo FAL",
           ha="center", fontsize=7, color=INK)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

    # (b) sorteo
    for i in range(T):
        b.vlines(i + 0.5, 0, 1, color=ORANGE if es_fal[i] else BLUE,
                 lw=1.5)
    b.set_xlim(0, T)
    b.set_ylim(0, 1)
    b.set_yticks([])
    b.set_xticks([0, t_fin, T])
    b.set_xticklabels(["0", r"$(1-\alpha)\,T$", r"$T$"])
    b.set_xlabel("Paso de entrenamiento $t$", fontsize=8)
    for s in ("top", "right", "left"):
        b.spines[s].set_visible(False)
    b.text(-0.01, 0.5, "Pérdida\nelegida", transform=b.transAxes,
           ha="right", va="center", fontsize=7.5)
    b.vlines([], 0, 1)
    # leyenda manual
    b.plot([], [], color=BLUE, lw=2, label="FCL")
    b.plot([], [], color=ORANGE, lw=2, label="FAL")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.85),
             ncol=2, frameon=False, fontsize=7.5,
             handlelength=1.4)
    a.set_title("(a)", loc="left", fontsize=8, pad=2)
    b.set_title("(b)", loc="left", fontsize=8, pad=2)
    guardar(fig, "facl_cronograma.pdf")


if __name__ == "__main__":
    r = fig_espectros()
    print({k: round(v, 5) for k, v in r.items()})
    fig_cronograma()
