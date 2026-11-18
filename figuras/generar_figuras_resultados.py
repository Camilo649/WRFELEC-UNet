# -*- coding: utf-8 -*-
"""
Figuras del capítulo de Resultados que salen de las estimaciones de
todo el conjunto de validación.

Uso (desde la carpeta de la monografía):
    python3 figuras/generar_figuras_resultados.py

Lee figuras/datos/resultados_validacion.npz, que se genera en la
notebook con figuras/celda_resultados_validacion.py (valor simulado
y estimaciones de los modelos, desnormalizados y redondeados, para
las 2.271 muestras de validación).

Genera:
    img/fotogramas_tormenta.pdf   seis pasos de tiempo de una
                                  tormenta de validación: WRF-ELEC
                                  y modelo final
    img/series_tormentas.pdf      actividad total por paso de tiempo
                                  de las cuatro tormentas
    img/csi_tormentas.pdf         CSI por paso de tiempo
    img/fss_matriz.pdf            FSS según umbral y vecindario
    img/distribucion_lightdens.pdf
                                  cantidad de columnas con cada
                                  valor de LIGHTDENS: WRF-ELEC y
                                  los dos modelos
y actualiza figuras/datos/comparacion_instante.npz (el paso de
tiempo de la figura de comparación; después hay que correr
figuras/generar_figuras_modelos.py). Las métricas se calculan con
figuras/metricas_resultados.py y se guardan en
figuras/datos/metricas_resultados.json (se recalculan si ese
archivo no existe).
"""
import datetime as dt
import json
import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from generar_figuras_instante import (  # noqa: E402
    limites, guardar, coma, SEQ, SEQ_RAYOS, INK, MUTED, BLUE,
    ORANGE)
from metricas_resultados import (  # noqa: E402
    metricas, csi_por_paso, UMBRALES, VECINDARIOS)

D = np.load(os.path.join(HERE, "datos", "resultados_validacion.npz"))
ARCHIVOS = [str(a) for a in D["archivos"]]
TIEMPOS = [dt.datetime.strptime(a[11:30], "%Y-%m-%d_%H-%M-%S")
           for a in ARCHIVOS]
REAL = D["y_real"]
TEMP = D["pred_temp"]
NOMBRE_FINAL = "Temp Ext Clip\nU-Net FACL"

# Modelos del archivo: clave -> (nombre, color, estilo de línea)
# La Ext Clip U-Net FACL se evalúa con los pesos de su última época
# (la 140).
MODELOS = {
    "pred_ext_ultima": ("Ext Clip U-Net FACL (época 140)", ORANGE,
                        "-"),
    "pred_temp": ("Temp Ext Clip U-Net FACL", BLUE, "-"),
}
RUTA_METRICAS = os.path.join(HERE, "datos", "metricas_resultados.json")


def tormentas():
    """Rangos [a, b) de índices de cada tormenta de validación (los
    pasos de tiempo de una tormenta están separados por 2 min)."""
    cortes = [0] + [k for k in range(1, len(TIEMPOS))
                    if (TIEMPOS[k] - TIEMPOS[k - 1]).total_seconds()
                    > 120] + [len(TIEMPOS)]
    return list(zip(cortes[:-1], cortes[1:]))


def cargar_metricas():
    """Métricas de todos los modelos (con la matriz de FSS)."""
    if os.path.exists(RUTA_METRICAS):
        with open(RUTA_METRICAS, encoding="utf8") as f:
            return json.load(f)
    cfg = tuple((u, v) for v in VECINDARIOS for u in UMBRALES)
    out = {}
    for clave in list(MODELOS) + ["pred_comb_mejor"]:
        print("Calculando métricas de", clave, "...")
        out[clave] = metricas(REAL, D[clave], fss_cfg=cfg)
    with open(RUTA_METRICAS, "w", encoding="utf8") as f:
        json.dump(out, f, indent=1)
    return out


def indice(texto):
    """Índice del paso de tiempo 'AAAA-MM-DD HH:MM' (UTC)."""
    t = dt.datetime.strptime(texto, "%Y-%m-%d %H:%M")
    return TIEMPOS.index(t)


# ---------------------------------------------------------------
# Fotogramas de una tormenta
# ---------------------------------------------------------------
FOTOGRAMAS = ["2018-12-14 01:22", "2018-12-14 03:22",
              "2018-12-14 05:22", "2018-12-14 07:22",
              "2018-12-14 09:22", "2018-12-14 11:22"]


def _mapa(ax, M, vmax):
    im = ax.imshow(M, origin="lower", cmap=SEQ_RAYOS, vmin=0,
                   vmax=vmax, interpolation="nearest")
    limites(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    for lado in ax.spines.values():
        lado.set_color(MUTED)
        lado.set_linewidth(0.5)
    return im


def fig_fotogramas():
    ids = [indice(t) for t in FOTOGRAMAS]
    vmax = 150.0
    fig = plt.figure(figsize=(6.6, 9.1))
    gs = fig.add_gridspec(
        5, 3, height_ratios=[1, 1, 1, 1, 0.045], wspace=0.05,
        hspace=0.12, left=0.07, right=0.995, top=0.975,
        bottom=0.045)
    im = None
    for bloque in range(2):               # tres pasos por bloque
        for c in range(3):
            k = ids[3 * bloque + c]
            ax_r = fig.add_subplot(gs[2 * bloque, c])
            ax_m = fig.add_subplot(gs[2 * bloque + 1, c])
            im = _mapa(ax_r, REAL[k], vmax)
            _mapa(ax_m, TEMP[k], vmax)
            ax_r.set_title(f"{TIEMPOS[k]:%H:%M} UTC", fontsize=8,
                           color=INK, pad=3)
            if c == 0:
                ax_r.set_ylabel("WRF-ELEC", fontsize=8, color=INK,
                                labelpad=4)
                ax_m.set_ylabel(NOMBRE_FINAL, fontsize=8, color=INK,
                                labelpad=4, linespacing=1.15)
    cax = fig.add_subplot(gs[4, :])
    pos = cax.get_position()
    cax.set_position([pos.x0 + 0.2 * pos.width, pos.y0 - 0.012,
                      0.6 * pos.width, pos.height])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.locator = mpl.ticker.MaxNLocator(7)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=7.5, length=2.5, width=0.4,
                      labelcolor=MUTED)
    cb.outline.set_visible(False)
    cb.set_label("LIGHTDENS", fontsize=8, color=MUTED, labelpad=2)
    guardar(fig, "fotogramas_tormenta.pdf")


# ---------------------------------------------------------------
# Actividad total y CSI por paso de tiempo
# ---------------------------------------------------------------
def _eje(ax):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(MUTED)
    ax.tick_params(labelsize=7, length=2.5, width=0.5,
                   labelcolor=MUTED, color=MUTED)


def _titulo_tormenta(ax, a, b, letra):
    ax.set_title(f"({letra}) {TIEMPOS[a]:%d/%m/%Y}, desde las "
                 f"{TIEMPOS[a]:%H:%M} UTC", fontsize=8, color=INK,
                 loc="left", pad=4)


def fig_series():
    total = lambda M: M.reshape(len(M), -1).sum(1, dtype=np.int64)
    s_real = total(REAL) / 1e3
    series = {k: total(D[k]) / 1e3 for k in MODELOS}
    # Opcional: Ext Clip U-Net FACL con los pesos de su mejor época
    # según L_val (la 8). La agrega figuras/celda_ext_epoca8.py al
    # archivo de resultados (clave pred_ext_epoca8).
    EXTRA_EPOCA8 = (total(D["pred_ext_epoca8"]) / 1e3
                    if "pred_ext_epoca8" in D.files else None)
    fig, axs = plt.subplots(2, 2, figsize=(6.8, 4.9),
                            gridspec_kw={"hspace": 0.5,
                                         "wspace": 0.2})
    for ax, (a, b), letra in zip(axs.ravel(), tormentas(), "abcd"):
        h = np.arange(b - a) * 2 / 60.0
        ax.plot(h, s_real[a:b], color=INK, lw=1.1, label="WRF-ELEC",
                zorder=3)
        for k, (nombre, color, ls) in MODELOS.items():
            ax.plot(h, series[k][a:b], color=color, lw=0.9, ls=ls,
                    label=nombre, zorder=2)
        if EXTRA_EPOCA8 is not None:
            ax.plot(h, EXTRA_EPOCA8[a:b], color=ORANGE, lw=0.9,
                    ls=(0, (3, 1.5)),
                    label="Ext Clip U-Net FACL (época 8)", zorder=2)
        _eje(ax)
        _titulo_tormenta(ax, a, b, letra)
        ax.set_xlim(0, h[-1])
        ax.set_ylim(0, None)
        ax.set_xlabel("Horas desde el inicio", fontsize=7.5,
                      color=MUTED)
    for ax in axs[:, 0]:
        ax.set_ylabel("Actividad total (miles de descargas)",
                      fontsize=7.5,
                      color=MUTED)
    hs, ls = axs[0, 0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="lower center", ncol=2, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.075),
               handlelength=2.6, columnspacing=2.0)
    guardar(fig, "series_tormentas.pdf")


def _media_movil(x, n):
    """Media móvil centrada de n pasos que ignora los NaN."""
    ok = ~np.isnan(x)
    num = np.convolve(np.where(ok, x, 0.0), np.ones(n), mode="same")
    den = np.convolve(ok.astype(float), np.ones(n), mode="same")
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


VENTANA_CSI = 15      # pasos de tiempo (30 minutos)


def fig_csi():
    claves = ["pred_ext_ultima", "pred_temp"]
    csi = {k: csi_por_paso(REAL, D[k]) for k in claves}
    s_real = REAL.reshape(len(REAL), -1).sum(1, dtype=np.int64) / 1e3
    fig, axs = plt.subplots(2, 2, figsize=(6.8, 4.9),
                            gridspec_kw={"hspace": 0.5,
                                         "wspace": 0.2})
    for ax, (a, b), letra in zip(axs.ravel(), tormentas(), "abcd"):
        h = np.arange(b - a) * 2 / 60.0
        ax2 = ax.twinx()
        ax2.fill_between(h, 0, s_real[a:b], color="#dcdcd8", lw=0,
                         zorder=0)
        ax2.set_ylim(0, s_real.max() * 1.02)
        ax2.set_yticks([])
        for lado in ax2.spines.values():
            lado.set_visible(False)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        for k in claves:
            nombre, color, ls = MODELOS[k]
            ax.plot(h, _media_movil(csi[k][a:b], VENTANA_CSI),
                    color=color, lw=0.9, ls=ls, label=nombre)
        _eje(ax)
        _titulo_tormenta(ax, a, b, letra)
        ax.set_xlim(0, h[-1])
        ax.set_ylim(0, 1)
        ax.set_xlabel("Horas desde el inicio", fontsize=7.5,
                      color=MUTED)
    for ax in axs[:, 0]:
        ax.set_ylabel("CSI", fontsize=7.5, color=MUTED)
    hs, ls = axs[0, 0].get_legend_handles_labels()
    hs.append(mpl.patches.Patch(color="#dcdcd8"))
    ls.append("Actividad total simulada")
    fig.legend(hs, ls, loc="lower center", ncol=3, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.04),
               handlelength=2.6, columnspacing=2.0)
    guardar(fig, "csi_tormentas.pdf")


# ---------------------------------------------------------------
# Matriz de FSS
# ---------------------------------------------------------------
def fig_fss():
    met = cargar_metricas()
    claves = ["pred_ext_ultima", "pred_temp"]
    titulos = ["(a) Ext Clip U-Net FACL",
               "(b) Temp Ext Clip U-Net FACL"]
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.75),
                            gridspec_kw={"wspace": 0.12})
    for ax, k, tit in zip(axs, claves, titulos):
        Mx = np.array([[met[k][f"FSS_u{u}_v{v}"] for u in UMBRALES]
                       for v in VECINDARIOS])
        im = ax.imshow(Mx, origin="lower", cmap=SEQ, vmin=0.4,
                       vmax=1.0, aspect="auto")
        for i in range(len(VECINDARIOS)):
            for j in range(len(UMBRALES)):
                ax.text(j, i, coma(Mx[i, j], 2), ha="center",
                        va="center", fontsize=8,
                        color="white" if Mx[i, j] > 0.78 else INK)
        ax.set_xticks(range(len(UMBRALES)))
        ax.set_xticklabels([str(u) for u in UMBRALES])
        ax.set_yticks(range(len(VECINDARIOS)))
        ax.set_yticklabels([f"{v}$\\times${v}"
                            for v in VECINDARIOS])
        ax.set_xticks(np.arange(-.5, len(UMBRALES)), minor=True)
        ax.set_yticks(np.arange(-.5, len(VECINDARIOS)), minor=True)
        ax.grid(which="minor", color="white", lw=1.4)
        ax.tick_params(which="both", length=0, labelsize=7.5,
                       labelcolor=MUTED)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(tit, fontsize=8, color=INK, loc="left", pad=4)
        ax.set_xlabel("Umbral de LIGHTDENS", fontsize=7.5,
                      color=MUTED)
    axs[0].set_ylabel("Vecindario (columnas)", fontsize=7.5,
                      color=MUTED)
    axs[1].set_yticklabels([])
    cb = fig.colorbar(im, ax=list(axs), fraction=0.035, pad=0.02,
                      ticks=[0.4, 0.6, 0.8, 1.0])
    cb.ax.set_yticklabels([coma(x, 1) for x in (0.4, 0.6, 0.8, 1.0)])
    cb.set_label("FSS", fontsize=7.5, color=MUTED)
    cb.ax.tick_params(labelsize=7.5, labelcolor=MUTED, length=0)
    cb.outline.set_visible(False)
    guardar(fig, "fss_matriz.pdf")


# ---------------------------------------------------------------
# Distribución de LIGHTDENS
# ---------------------------------------------------------------
def _conteo(M):
    """Cantidad de columnas con cada valor de 1 a 150 (NaN donde no
    hay ninguna, para que la curva se corte en escala log)."""
    c = np.bincount(M.ravel(), minlength=151)[1:151].astype(float)
    c[c == 0] = np.nan
    return c


def fig_distribucion():
    """Un solo panel: WRF-ELEC y los modelos de MODELOS."""
    v = np.arange(1, 151)
    fig, ax = plt.subplots(figsize=(6.0, 3.1))
    ax.plot(v, _conteo(REAL), drawstyle="steps-mid", color=INK,
            lw=0.7, zorder=4, label="WRF-ELEC")
    for clave, (nombre, color, _) in MODELOS.items():
        ax.plot(v, _conteo(D[clave]), drawstyle="steps-mid",
                color=color, lw=0.9, zorder=3,
                label=nombre.replace(" (época 140)", ""))
    ax.set_yscale("log")
    ax.set_xlim(0, 153)
    ax.set_ylim(0.7, 2e6)
    ax.set_xticks([0, 25, 50, 75, 100, 125, 150])
    _eje(ax)
    ax.tick_params(which="minor", length=1.5, width=0.4,
                   color=MUTED)
    ax.set_xlabel("LIGHTDENS (descargas por columna)", fontsize=7.5,
                  color=MUTED)
    ax.set_ylabel("Cantidad de columnas", fontsize=7.5, color=MUTED)
    ax.legend(loc="upper right", bbox_to_anchor=(0.97, 1.0),
              fontsize=7.5, frameon=False, handlelength=2.2)
    guardar(fig, "distribucion_lightdens.pdf")


# ---------------------------------------------------------------
# Paso de tiempo de la figura de comparación
# ---------------------------------------------------------------
ARCHIVO_COMPARACION = "wrfout_d01_2018-12-14_01-00-00.npz"


def actualizar_comparacion():
    """Reescribe datos/comparacion_instante.npz con las estimaciones
    del archivo de resultados (pesos de la mejor época de cada
    modelo, salvo la Ext Clip U-Net FACL, con los de su última
    época). La figura se arma con generar_figuras_modelos.py."""
    i = ARCHIVOS.index(ARCHIVO_COMPARACION)
    nombres = ["Ext U-Net Comb V2.1",
               "Ext Clip U-Net FACL (Optimized)",
               "Temp Ext Clip U-Net FACL"]
    claves = ["pred_comb_mejor", "pred_ext_ultima", "pred_temp"]
    preds = np.stack([D[k][i].astype("float32") for k in claves])
    real = REAL[i].astype("float32")
    csi = [float(csi_por_paso(REAL[i:i + 1], D[k][i:i + 1])[0])
           for k in claves]
    ruta = os.path.join(HERE, "datos", "comparacion_instante.npz")
    np.savez(ruta, nombres=np.array(nombres), y_real=real,
             preds=preds, csi=np.array(csi),
             archivo=ARCHIVO_COMPARACION)
    print("OK", ruta)
    for n, k in zip(nombres, claves):
        p = D[k][i]
        print(f"   {n}: {int((p > 0).sum())} columnas, total "
              f"{int(p.sum())}, máximo {p.max()}")


if __name__ == "__main__":
    fig_fotogramas()
    fig_series()
    fig_csi()
    fig_fss()
    fig_distribucion()
    actualizar_comparacion()
