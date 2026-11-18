# -*- coding: utf-8 -*-
"""
Serie de descargas cada 2 minutos: LMA y WRF-ELEC (Fig. 4.x).

El LMA (RELAMPAGO) solo registra descargas a menos de 100 km de su
centro (-31,7 / -64,1); el dominio del WRF abarca unos 600 km de
lado. Para que la comparación sea directa, la serie del WRF se
calcula (a) en todo el dominio y (b) solo dentro de ese radio.

Es autocontenido (no importa nada propio). Uso en JupyterHub: pegar
todo el archivo en una celda de Datasets.ipynb, donde ya existe
`eventos` (celda "Corridas continuas"), y agregar al final:

    serie_wrf_en_radio(eventos)      # escribe serie_wrf_radio_25ene.csv

Después pasame serie_wrf_radio_25ene.csv.
"""
import multiprocessing as mp
import numpy as np
import pandas as pd
from netCDF4 import Dataset

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x

import os
import re
from datetime import datetime

PAT = re.compile(
    r"wrfout_d01_(\d{4}-\d{2}-\d{2})_(\d{2})[:\-](\d{2})[:\-](\d{2})")


def t_nombre(ruta):
    """Fecha y hora de la salida, tomadas del nombre del archivo."""
    m = PAT.search(os.path.basename(str(ruta)))
    return datetime.strptime(
        f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}",
        "%Y-%m-%d %H:%M:%S")


LAT0, LON0, RADIO_KM = -31.7, -64.1, 100.0     # centro y radio del LMA
DIAS = ("2019-01-25",)                          # día de mayor actividad del LMA
N_PROC = 16


def _dist_km(lat, lon):
    r = 6371.0
    p1, p2 = np.radians(lat), np.radians(LAT0)
    dl = np.radians(lon - LON0)
    a = (np.sin((p2 - p1) / 2) ** 2 +
         np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2)
    return 2 * r * np.arcsin(np.sqrt(a))


def _leer(ds, nombre):
    if nombre not in ds.variables:
        return None
    v = np.array(ds.variables[nombre][:], dtype="float64")
    return np.squeeze(v)          # (Time, y, x) con un solo tiempo


def _fila(f):
    with Dataset(f) as ds:
        lat = _leer(ds, "XLAT")
        lon = _leer(ds, "XLONG")
        lg = _leer(ds, "LIGHT")
        ld = _leer(ds, "LIGHTDENS")
    # las variables escalonadas pueden tener 300 en alguna dimensión
    n = min(lat.shape[0], lg.shape[0])
    m = min(lat.shape[1], lg.shape[1])
    lat, lon, lg, ld = (a[:n, :m] for a in (lat, lon, lg, ld))
    dentro = _dist_km(lat, lon) <= RADIO_KM
    return (t_nombre(f), lg.sum(), lg[dentro].sum(),
            ld.sum(), ld[dentro].sum(),
            np.minimum(ld, 150)[dentro].sum(), int(dentro.sum()))


def serie_wrf_en_radio(eventos, ruta_csv="serie_wrf_radio_25ene.csv",
                       dias=DIAS, n_proc=N_PROC):
    archivos = sorted({f for fs in eventos.values() for f in fs
                       if f"{t_nombre(f):%Y-%m-%d}" in dias},
                      key=t_nombre)
    print(len(archivos), "salidas en", dias)
    with mp.get_context("fork").Pool(n_proc) as pool:
        filas = list(tqdm(pool.imap(_fila, archivos, chunksize=4),
                          total=len(archivos)))
    df = pd.DataFrame(filas, columns=[
        "t", "light_dominio", "light_radio", "lightdens_dominio",
        "lightdens_radio", "lightdens_radio_sat150", "n_pix_radio"])
    df.to_csv(ruta_csv, index=False)
    for dia in dias:
        d = df[df.t.dt.strftime("%Y-%m-%d") == dia]
        if len(d):
            print(dia, "| máx LIGHT dominio", d.light_dominio.max(),
                  "| máx LIGHT en el radio", d.light_radio.max(),
                  "| píxeles en el radio", int(d.n_pix_radio.iloc[0]))
    return df
