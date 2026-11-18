# -*- coding: utf-8 -*-
"""
Celda para JupyterHub (Datasets.ipynb, donde existe `eventos`):
guarda la máscara del círculo de 100 km del LMA sobre los mapas
recortados de 297x297, para sumar las descargas predichas por la red
dentro del radio. Misma definición que serie_lma_wrf.py.
Escribe mascara_radio_lma_297.npy; pasame ese archivo.
"""
import glob
import numpy as np
from netCDF4 import Dataset

LAT0, LON0, RADIO_KM = -31.7, -64.1, 100.0
# cualquier wrfout del dominio sirve (la grilla es fija)
f = sorted(glob.glob("RUTA_A_LOS_WRFOUT/wrfout_d01_2018-12-14_*"))[0]
with Dataset(f) as ds:
    lat = np.squeeze(np.array(ds.variables["XLAT"][:], dtype="float64"))
    lon = np.squeeze(np.array(ds.variables["XLONG"][:], dtype="float64"))
r = 6371.0
p1, p2 = np.radians(lat), np.radians(LAT0)
a = (np.sin((p2 - p1) / 2) ** 2 +
     np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon - LON0) / 2) ** 2)
dist = 2 * r * np.arcsin(np.sqrt(a))
dentro = dist <= RADIO_KM
print("grilla", lat.shape, "| columnas dentro:", int(dentro.sum()))
mask297 = dentro[1:-1, 1:-1]            # mismo recorte que los mapas
print("mascara 297:", mask297.shape, int(mask297.sum()))
np.save("mascara_radio_lma_297.npy", mask297)
