# -*- coding: utf-8 -*-
"""
Figura del LMA (img/lma_vs_wrf.pdf): descargas cada 2 minutos el
25/01/2019 (el día de mayor actividad del LMA), observadas por el
LMA y simuladas por el WRF-ELEC.

Uso:  python3 figuras/generar_figura_lma.py
Lee figuras/datos/lma_serie_2min.csv (lma_serie_2min.py, sale de
lma_claude.csv) y figuras/datos/serie_wrf_radio*.csv
(serie_lma_wrf.py).
"""
import glob
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif",
                   "DejaVu Serif"],
    "font.size": 8, "axes.linewidth": 0.5, "pdf.fonttype": 42,
    "savefig.dpi": 300,
})
HERE = os.path.dirname(os.path.abspath(__file__))
DATOS = os.path.join(HERE, "datos")
IMG = os.path.join(HERE, "..", "img")
DIA = "2019-01-25"
AZUL, NARANJA, INK, MUTED = "#2f6ca3", "#c8553d", "#2b2b2b", "#6b6b6b"


def coma(x, pos=None):
    return f"{x:,.0f}".replace(",", ".")


lma = pd.read_csv(os.path.join(DATOS, "lma_serie_2min.csv"))
lma = lma[lma.fecha == DIA]
w = pd.concat([pd.read_csv(f, parse_dates=["t"]) for f in
               sorted(glob.glob(os.path.join(
                   DATOS, "serie_wrf_radio*.csv")))])
w = w[w.t.dt.strftime("%Y-%m-%d") == DIA].drop_duplicates("t")
if w.empty:
    raise SystemExit(f"Falta la serie del WRF-ELEC para {DIA}: correr "
                     "serie_lma_wrf.py y copiar el CSV en datos/.")
w = w.sort_values("t")
h = w.t.dt.hour + w.t.dt.minute / 60
dd = f"{DIA[8:]}/{DIA[5:7]}/{DIA[:4]}"
print("LMA: máx cada 2 min =", lma.descargas.max(),
      "| total =", lma.descargas.sum())
print("WRF-ELEC (radio): máx LIGHT cada 2 min =",
      w.light_radio.max(), "| máx restringido*K =",
      (w.lightdens_radio_sat150 * (w.light_radio.sum()
       / w.lightdens_radio.sum())).max())

# LIGHTDENS no es un conteo de descargas: para ponerla en las mismas
# unidades que LIGHT se la escala con el cociente entre las sumas de
# LIGHT y de LIGHTDENS (sin saturar) de todo el día, dentro del radio.
K = w.light_radio.sum() / w.lightdens_radio.sum()
print("factor LIGHT/LIGHTDENS =", K)
fig, (ax, bx) = plt.subplots(2, 1, figsize=(6.6, 5.0), sharex=True,
                             sharey=True, gridspec_kw={"hspace": 0.18})
for a_, serie, nom in ((ax, w.light_radio, "WRF-ELEC (simulado)"),
                       (bx, w.lightdens_radio_sat150 * K,
                        "WRF-ELEC (simulado, restringido)")):
    a_.step(lma.hora_utc, lma.descargas, where="post", color=AZUL,
            lw=1.0, label="LMA (observado)")
    a_.step(h, serie, where="post", color=NARANJA, lw=1.0, label=nom)
    a_.set_ylabel("Descargas cada 2 minutos", color=INK)
    a_.legend(frameon=False, fontsize=7.5, loc="upper left")
    for lado in ("top", "right"):
        a_.spines[lado].set_visible(False)
    a_.tick_params(width=0.5, labelcolor=MUTED)
ax.set_ylim(0, 1.06 * max(w.light_radio.max(), lma.descargas.max()))
ax.set_xlim(15, 24)
ax.set_xticks(range(15, 25))
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(coma))
ax.set_title(r"$\mathbf{(a)}$ Valores originales", loc="left",
             fontsize=8.5, color=INK)
bx.set_title(r"$\mathbf{(b)}$ Con la restricción a 150 descargas por columna aplicada",
             loc="left", fontsize=8.5, color=INK)
bx.set_xlabel(f"Hora (UTC) del {dd}", color=INK)
os.makedirs(IMG, exist_ok=True)
fig.savefig(os.path.join(IMG, "lma_vs_wrf.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(IMG, "lma_vs_wrf.png"), bbox_inches="tight")
print("OK")
