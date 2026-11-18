"""
Diagrama de la U-Net (UNet_WRF) para la monografía.

Reemplazo "drop-in" del `draw_unet` de `modulos/unet_diagram.py`:
mantiene los mismos argumentos (in_channels, n_niveles,
n_filtros_base, kernel_sizes, X_sample, y_sample, y_pred_sample,
var_names, var_cmaps, target_cmap, lon_min/lon_max/lat_min/lat_max,
out_path) y agrega soporte para el dataset temporal de 9 canales:
las variables se acomodan en una grilla (filas = instantes/diferencias,
columnas = variables) en lugar de una pila de mapas superpuestos.

Ejemplo con datos reales (en la notebook, 9 canales):

    from modulos.unet_diagram import draw_unet   # este archivo
    draw_unet(
        in_channels=9, n_niveles=3, n_filtros_base=64,
        kernel_sizes=[5, 3, 3],
        X_sample=X_muestra,            # array (9, H, W) o tensor
        y_pred_sample=y_pred,          # opcional, array (H, W)
        y_sample=None,                 # opcional, el objetivo real
        var_names=["mdbz", "hielo", "ctt"],
        var_cmaps=["jet", "Blues", "RdBu_r"],
        target_cmap="hot_r",
        lon_min=lon_min, lon_max=lon_max,
        lat_min=lat_min, lat_max=lat_max,
        out_path="unet_arquitectura.pdf",
    )

Sin X_sample (o sin y_pred_sample) dibuja casilleros vacíos: sirve
para generar el esquema sin datos. Los contornos de costas/fronteras
se dibujan solo si hay cartopy y se pasan lon/lat.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle, FancyArrowPatch

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif",
                   "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
})

INK = "#222222"
MUTED = "#8a8a85"
LGRAY = "#ececea"
BLK_FRONT, BLK_TOP, BLK_SIDE = "#aecbe8", "#8fb1db", "#6f98c4"
C_DOWN, C_UP, C_SKIP = "#6aa84f", "#d9a441", "#d94f4f"

GRUPOS_TEMPORALES = [r"$X(t)$", r"$X(t)-X(t-1)$",
                     r"$X(t-1)-X(t-2)$"]


# ---------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------
def _np(a):
    if a is None:
        return None
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=float)


def _contornos(lon_min, lon_max, lat_min, lat_max):
    """Polilíneas (lon, lat) de costas y fronteras, o [] si no hay
    cartopy / datos de Natural Earth."""
    if None in (lon_min, lon_max, lat_min, lat_max):
        return []
    try:
        import cartopy.feature as cfeature
        feats = [cfeature.COASTLINE.with_scale("50m"),
                 cfeature.BORDERS.with_scale("50m")]
        lineas = []
        for f in feats:
            for g in f.geometries():
                partes = g.geoms if hasattr(g, "geoms") else [g]
                for p in partes:
                    xs, ys = p.xy
                    lineas.append((np.asarray(xs), np.asarray(ys)))
        return lineas
    except Exception:
        return []


def _tile(ax, x0, y0, s, datos=None, cmap="viridis", vmin=None,
          vmax=None, contornos=(), bbox=None, relleno=LGRAY,
          origen_lower=True):
    """Casillero cuadrado de lado s con su esquina inferior
    izquierda en (x0, y0). Con `datos`, dibuja el mapa."""
    if datos is None:
        ax.add_patch(Rectangle((x0, y0), s, s, fc=relleno, ec=INK,
                               lw=0.5, zorder=3))
        return
    ax.imshow(datos, cmap=cmap, vmin=vmin, vmax=vmax,
              extent=(x0, x0 + s, y0, y0 + s), zorder=3,
              origin="lower" if origen_lower else "upper",
              interpolation="nearest", aspect="auto")
    if bbox is not None and len(contornos):
        lo0, lo1, la0, la1 = bbox
        for xs, ys in contornos:
            ok = (xs >= lo0) & (xs <= lo1) & (ys >= la0) & (ys <= la1)
            if ok.sum() < 2:
                continue
            X = np.where(ok, x0 + (xs - lo0) / (lo1 - lo0) * s,
                         np.nan)
            Y = np.where(ok, y0 + (ys - la0) / (la1 - la0) * s,
                         np.nan)
            ax.plot(X, Y, color="k", lw=0.25, zorder=4)
    ax.add_patch(Rectangle((x0, y0), s, s, fc="none", ec=INK,
                           lw=0.5, zorder=5))


def _bloque(ax, x, yc, w, h, dx3, dy3):
    """Prisma (cara frontal + tapa + lado) con centro vertical yc."""
    yb, yt = yc - h / 2, yc + h / 2
    ax.add_patch(Rectangle((x, yb), w, h, fc=BLK_FRONT, ec=INK,
                           lw=0.5, zorder=3))
    ax.add_patch(Polygon([(x, yt), (x + w, yt),
                          (x + w + dx3, yt + dy3),
                          (x + dx3, yt + dy3)],
                         fc=BLK_TOP, ec=INK, lw=0.5, zorder=3))
    ax.add_patch(Polygon([(x + w, yb), (x + w + dx3, yb + dy3),
                          (x + w + dx3, yt + dy3), (x + w, yt)],
                         fc=BLK_SIDE, ec=INK, lw=0.5, zorder=3))


def _flecha(ax, p, q, color, ls="-", lw=1.0, ms=7, z=2):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=ms, lw=lw,
        linestyle=ls, color=color, shrinkA=0, shrinkB=0, zorder=z))


# ---------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------
def draw_unet(in_channels=3, n_niveles=3, n_filtros_base=64,
              kernel_sizes=None, X_sample=None, y_sample=None,
              y_pred_sample=None, var_names=None, var_cmaps=None,
              target_cmap="hot_r", lon_min=None, lon_max=None,
              lat_min=None, lat_max=None,
              out_path="unet_arquitectura.png",
              grupos=None, diff_cmap=None, diff_cero=None, titulo=None,
              figsize=(7.2, 3.9), fontsize=7, dpi=300,
              etiqueta_salida="salida\n(predicción)",
              etiqueta_objetivo="valor real",
              mostrar_leyenda=True, mostrar=False):
    L = int(n_niveles)
    if kernel_sizes is None:
        kernel_sizes = [3] * L
    assert len(kernel_sizes) == L, "kernel_sizes debe tener n_niveles"
    X = _np(X_sample)
    yp = _np(y_pred_sample)
    yt = _np(y_sample)

    # --- grilla de entrada: columnas = variables, filas = grupos ---
    nv = len(var_names) if var_names else in_channels
    if in_channels % nv != 0:
        nv = in_channels
    filas = in_channels // nv
    if var_names is None or len(var_names) != nv:
        var_names = [f"canal {i + 1}" for i in range(nv)]
    if var_cmaps is None:
        var_cmaps = ["viridis"] * nv
    if grupos is None and filas == len(GRUPOS_TEMPORALES):
        grupos = GRUPOS_TEMPORALES
    if X is not None:
        assert X.shape[0] == in_channels, (
            f"X_sample tiene {X.shape[0]} canales y se pidieron "
            f"{in_channels}")

    contornos = _contornos(lon_min, lon_max, lat_min, lat_max)
    bbox = (lon_min, lon_max, lat_min, lat_max) \
        if contornos else None

    # --- geometría (unidades arbitrarias) ---
    t, gap = 0.95, 0.10                       # casillero y separación
    grid_w = nv * t + (nv - 1) * gap
    grid_h = filas * t + (filas - 1) * gap
    h0 = max(2.4, grid_h)
    x_grid = 2.15 if filas > 1 else 0.3
    x_start = x_grid + grid_w + 1.55
    dx = 2.2
    esc = 0.72                                # alto por nivel
    gap_v = 1.0                               # hueco entre niveles

    def canales(i):                           # canales del nivel i
        return n_filtros_base * 2 ** i

    def ancho(c):                             # espesor del prisma
        return 0.50 * (c / n_filtros_base) ** 0.33

    dx3, dy3 = 0.16, 0.16
    sep = 0.12

    # niveles 0..L-1 = codificador / decodificador, L = cuello
    alto = [h0 * esc ** i for i in range(L + 1)]
    yc = [0.0]
    for i in range(L):
        yc.append(yc[-1] - (alto[i] + alto[i + 1]) / 2 - gap_v)
    x_enc = [x_start + i * dx for i in range(L)]
    x_bot = x_start + L * dx
    x_dec = [x_start + (2 * L - i) * dx for i in range(L)]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- entrada ------------------------------------------------
    y_top_grid = h0 / 2
    for r in range(filas):
        for c in range(nv):
            idx = r * nv + c
            x0 = x_grid + c * (t + gap)
            y0 = y_top_grid - (r + 1) * t - r * gap
            if X is None:
                _tile(ax, x0, y0, t)
            else:
                cm = var_cmaps[c]
                vmin, vmax = X[idx].min(), X[idx].max()
                if r > 0 and diff_cmap is not None:
                    cm = diff_cmap
                    if diff_cero is not None:
                        # centro del colormap en "sin cambio" (valor
                        # normalizado de una diferencia nula)
                        z = diff_cero[c]
                        rr = np.abs(X[idx] - z).max()
                        vmin, vmax = z - rr, z + rr
                _tile(ax, x0, y0, t, X[idx], cm, vmin, vmax,
                      contornos, bbox)
    for c, nom in enumerate(var_names):
        ax.text(x_grid + c * (t + gap) + t / 2, y_top_grid + 0.12,
                nom.upper(), ha="center", va="bottom",
                fontsize=fontsize - 1.3)
    if grupos is not None:
        for r in range(filas):
            ax.text(x_grid - 0.12,
                    y_top_grid - r * (t + gap) - t / 2, grupos[r],
                    ha="right", va="center", fontsize=fontsize - 1)
    # flecha entrada -> primer bloque
    xi = x_grid + grid_w + 0.12
    _flecha(ax, (xi, 0), (x_enc[0] - 0.25, 0), INK, lw=0.8)
    ax.text((xi + x_enc[0] - 0.25) / 2, 0.14,
            f"{in_channels}\ncanales", ha="center", va="bottom",
            fontsize=fontsize - 1, color=INK)

    # ---- codificador, cuello, decodificador ----------------------
    def _doble(x, y, h, c):
        w = ancho(c)
        _bloque(ax, x, y, w, h, dx3, dy3)
        _bloque(ax, x + w + sep, y, w, h, dx3, dy3)
        return x, x + 2 * w + sep + dx3          # xl, xr

    ext = {}
    for i in range(L):
        c = canales(i)
        ext[("e", i)] = _doble(x_enc[i], yc[i], alto[i], c)
        ext[("d", i)] = _doble(x_dec[i], yc[i], alto[i], c)
        for tipo in ("e", "d"):
            xl, xr = ext[(tipo, i)]
            ax.text((xl + xr) / 2, yc[i] + alto[i] / 2 + dy3 + 0.08,
                    f"{c}\n$k={kernel_sizes[i]}$", ha="center",
                    va="bottom", fontsize=fontsize - 1)
    cb = canales(L)
    ext[("b", L)] = _doble(x_bot, yc[L], alto[L], cb)
    xl, xr = ext[("b", L)]
    ax.text((xl + xr) / 2, yc[L] - alto[L] / 2 - 0.12,
            f"{cb}\ncuello de botella\n$k={kernel_sizes[-1]}$",
            ha="center", va="top", fontsize=fontsize - 1)

    def centro(clave):
        xl, xr = ext[clave]
        return (xl + xr) / 2

    # ---- flechas ------------------------------------------------
    for i in range(L):
        sig = ("e", i + 1) if i + 1 < L else ("b", L)
        # reducción (max pooling): abajo-derecha
        p = (ext[("e", i)][1] - 0.15, yc[i] - alto[i] / 2 - 0.06)
        q = (ext[sig][0] + 0.2,
             yc[i + 1] + alto[i + 1] / 2 + dy3 + 0.1)
        if i + 1 == L:        # el cuello lleva su rótulo abajo
            q = (ext[sig][0] + 0.2,
                 yc[i + 1] + alto[i + 1] / 2 + dy3 + 0.1)
        _flecha(ax, p, q, C_DOWN, lw=1.2)
        # ampliación (bilineal): arriba-derecha
        sig_d = ("d", i + 1) if i + 1 < L else ("b", L)
        p = (ext[sig_d][1] - 0.2,
             yc[i + 1] + alto[i + 1] / 2 + dy3 + 0.1)
        q = (ext[("d", i)][0] + 0.15, yc[i] - alto[i] / 2 - 0.06)
        _flecha(ax, p, q, C_UP, lw=1.2)
        # conexión de salto: encoder i -> decoder i
        _flecha(ax, (ext[("e", i)][1] + 0.1, yc[i]),
                (ext[("d", i)][0] - 0.1, yc[i]), C_SKIP,
                ls=(0, (4, 2)), lw=0.9, z=1)

    # ---- salida -------------------------------------------------
    s_out = 1.3
    xo = ext[("d", 0)][1] + 1.35
    _flecha(ax, (ext[("d", 0)][1] + 0.1, 0), (xo - 0.12, 0), INK,
            lw=0.8)
    ax.text((ext[("d", 0)][1] + xo) / 2, 0.14, "conv\n$1\\times1$",
            ha="center", va="bottom", fontsize=fontsize - 1)
    y_out = -s_out / 2
    if yp is not None:
        vmax = np.nanmax([yp.max(), yt.max() if yt is not None
                          else yp.max()])
        _tile(ax, xo, y_out, s_out, yp, target_cmap, 0, vmax,
              contornos, bbox)
    else:
        _tile(ax, xo, y_out, s_out)
    ax.text(xo + s_out / 2, y_out - 0.12, etiqueta_salida,
            ha="center", va="top", fontsize=fontsize - 1)
    if yt is not None:
        y2 = y_out - 1.0 - s_out
        vmax = np.nanmax([yt.max(), yp.max() if yp is not None
                          else yt.max()])
        _tile(ax, xo, y2, s_out, yt, target_cmap, 0, vmax,
              contornos, bbox)
        ax.text(xo + s_out / 2, y2 - 0.12, etiqueta_objetivo,
                ha="center", va="top", fontsize=fontsize - 1)

    # ---- leyenda ------------------------------------------------
    if mostrar_leyenda:
        hs = [Patch(fc=BLK_FRONT, ec=INK, lw=0.5,
                    label="Bloque: 2 $\\times$ (conv. + ReLU)"),
              Line2D([0], [0], color=C_DOWN, lw=1.3,
                     label="Reducción (max pooling $2\\times2$)"),
              Line2D([0], [0], color=C_UP, lw=1.3,
                     label="Ampliación (interpolación bilineal)"),
              Line2D([0], [0], color=C_SKIP, lw=1.1, ls=(0, (4, 2)),
                     label="Conexión de salto (concatenación)")]
        ax.legend(handles=hs, loc="lower left",
                  bbox_to_anchor=(0.0, 0.0), frameon=False,
                  fontsize=fontsize - 0.5, handlelength=2.0,
                  borderaxespad=0.0, labelspacing=0.5)

    if titulo:
        ax.set_title(titulo, fontsize=fontsize + 1)
    ax.autoscale_view()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    if out_path.lower().endswith(".pdf"):
        fig.savefig(out_path[:-4] + ".png", dpi=200,
                    bbox_inches="tight", pad_inches=0.03)
    if mostrar:
        plt.show()
    else:
        plt.close(fig)
    return out_path


if __name__ == "__main__":
    # Esquema sin datos (casilleros vacíos), 3 niveles, 9 canales.
    draw_unet(in_channels=9, n_niveles=3, n_filtros_base=64,
              kernel_sizes=[5, 3, 3],
              var_names=["mdbz", "hielo", "ctt"],
              out_path="unet_arquitectura_esquema.pdf")
