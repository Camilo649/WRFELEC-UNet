# -*- coding: utf-8 -*-
"""Figura img/saturacion_lma.pdf: forma temporal de la actividad el
14/12/2018 (LMA, LIGHTDENS sin saturar y saturada en 150, dentro del
radio de 100 km del LMA), cada curva dividida por su máximo.
Uso: python3 figuras/generar_figura_saturacion.py"""
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 8, "axes.linewidth": 0.5, "pdf.fonttype": 42,
    "savefig.dpi": 300})
HERE = os.path.dirname(os.path.abspath(__file__))
DIA = "2018-12-14"
AZUL, NARANJA, VERDE, INK, MUTED = ("#2f6ca3", "#c8553d", "#6b6b6b",
                                    "#2b2b2b", "#6b6b6b")
lma = pd.read_csv(os.path.join(HERE, "datos", "lma_serie_2min.csv"))
lma = lma[lma.fecha == DIA]
w = pd.read_csv(os.path.join(HERE, "datos", "serie_wrf_radio.csv"),
                parse_dates=["t"])
w = w[w.t.dt.strftime("%Y-%m-%d") == DIA]
h = w.t.dt.hour + w.t.dt.minute / 60
fig, ax = plt.subplots(figsize=(6.6, 2.7))
ax.step(lma.hora_utc, lma.descargas / lma.descargas.max(), where="post",
        color=AZUL, lw=1.1, label="LMA (observado)")
ax.step(h, w.lightdens_radio / w.lightdens_radio.max(), where="post",
        color=NARANJA, lw=1.0, ls=(0, (3, 1.5)),
        label=r"WRF-ELEC, \texttt{LIGHTDENS} original".replace(
            r"\texttt{LIGHTDENS}", "LIGHTDENS"))
ax.step(h, w.lightdens_radio_sat150 / w.lightdens_radio_sat150.max(),
        where="post", color=NARANJA, lw=1.1,
        label="WRF-ELEC, LIGHTDENS saturada en 150")
ax.set_xlim(0, 12)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Hora (UTC) del 14/12/2018", color=INK)
ax.set_ylabel("Valor / máximo de cada curva", color=INK)
for lado in ("top", "right"):
    ax.spines[lado].set_visible(False)
ax.tick_params(width=0.5, labelcolor=MUTED)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
fig.savefig(os.path.join(HERE, "..", "img", "saturacion_lma.pdf"),
            bbox_inches="tight")
fig.savefig(os.path.join(HERE, "..", "img", "saturacion_lma.png"),
            bbox_inches="tight")
print("OK")
