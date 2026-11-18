# -*- coding: utf-8 -*-
"""
Figura (img/lma_vs_temp.pdf): descargas cada 2 minutos el 14/12/2018
(día de validación con mayor actividad del LMA, que es el segundo
día más activo del LMA; el primero, 25/01/2019, es de entrenamiento),
observadas por el LMA y estimadas por la Temp Ext Clip U-Net FACL,
junto con las simuladas por el WRF-ELEC.

Necesita figuras/datos/mascara_radio_lma_297.npy
(mascara_radio_lma.py) y resultados_validacion.npz.
Uso:  python3 figuras/generar_figura_lma_temp.py
"""
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
DIA = "2018-12-14"
AZUL, NARANJA, VERDE, INK, MUTED = ("#2f6ca3", "#c8553d", "#3a8a4f",
                                    "#2b2b2b", "#6b6b6b")
def coma(x, pos=None):
    return f"{x:,.0f}".replace(",", ".")


mask = np.load(os.path.join(DATOS, "mascara_radio_lma_297.npy"))
d = np.load(os.path.join(DATOS, "resultados_validacion.npz"),
            allow_pickle=True)
arch = d["archivos"]
sel = np.array([a[11:21] == DIA for a in arch])
t = pd.to_datetime([a[11:21] + " " + a[22:30].replace("-", ":")
                    for a in arch[sel]])
h = np.asarray(t.hour + t.minute / 60)
pred = d["pred_temp"][sel][:, mask].astype("float64").sum(1)
real = d["y_real"][sel][:, mask].astype("float64").sum(1)

lma = pd.read_csv(os.path.join(DATOS, "lma_serie_2min.csv"))
lma = lma[lma.fecha == DIA]
w = pd.read_csv(os.path.join(DATOS, "serie_wrf_radio.csv"),
                parse_dates=["t"])
w = w[w.t.dt.strftime("%Y-%m-%d") == DIA].sort_values("t")
hw = np.asarray(w.t.dt.hour + w.t.dt.minute / 60)
# misma conversión que en la figura del LMA: LIGHTDENS (restringida
# en 150) a descargas con el cociente de las sumas de LIGHT y de
# LIGHTDENS del día, dentro del radio.
K = w.light_radio.sum() / w.lightdens_radio.sum()
print("K =", K, "| real (máscara) vs serie WRF restringida:",
      real.sum(), w.lightdens_radio_sat150.sum())
print("LMA: total", lma.descargas.sum(), "máx", lma.descargas.max())
print("Temp*K: total", pred.sum() * K, "máx", (pred * K).max())
print("WRF-ELEC*K: total", w.lightdens_radio_sat150.sum() * K)

fig, ax = plt.subplots(figsize=(6.6, 3.0))
ax.step(lma.hora_utc, lma.descargas, where="post", color=AZUL,
        lw=1.0, label="LMA (observado)")
ax.step(hw, w.lightdens_radio_sat150 * K, where="post", color=NARANJA,
        lw=1.0, label="WRF-ELEC (simulado, restringido)")
ax.step(h, pred * K, where="post", color=VERDE, lw=1.0,
        label="Temp Ext Clip U-Net FACL (estimado)")
ax.set_xlim(0, 8)
ax.set_ylim(0, 2100)
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(coma))
ax.set_xticks(range(0, 9))
ax.set_xlabel(f"Hora (UTC) del {DIA[8:]}/{DIA[5:7]}/{DIA[:4]}",
              color=INK)
ax.set_ylabel("Descargas cada 2 minutos", color=INK)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
for lado in ("top", "right"):
    ax.spines[lado].set_visible(False)
ax.tick_params(width=0.5, labelcolor=MUTED)
os.makedirs(IMG, exist_ok=True)
fig.savefig(os.path.join(IMG, "lma_vs_temp.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(IMG, "lma_vs_temp.png"), bbox_inches="tight")
print("OK")
