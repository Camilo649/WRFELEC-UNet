# -*- coding: utf-8 -*-
"""
Opciones eléctricas del namelist, leídas de los atributos globales
de un wrfout, y descripción de LIGHTDENS/LIGHT guardada por el
modelo. Pegar en una celda de Datasets.ipynb (donde ya existe
`eventos`) y pasarme lo que imprime.
"""
import numpy as np
from netCDF4 import Dataset

ruta = eventos[sorted(eventos)[0]][0]
with Dataset(ruta) as ds:
    print("== Atributos globales (NSSL / ELEC / LIGHT) ==")
    for k in sorted(ds.ncattrs()):
        if any(s in k.upper() for s in ("NSSL", "ELEC", "LIGHT",
                                        "LTNG", "DISCH", "BRK")):
            print(f"{k:28s} {np.asarray(getattr(ds, k)).tolist()}")
    print("\n== Variables eléctricas ==")
    for v in ("LIGHTDENS", "LIGHT", "LIGHTDIS", "INDUC", "NONINDUC"):
        if v in ds.variables:
            x = ds.variables[v]
            print(v, "|", getattr(x, "description", ""), "|",
                  getattr(x, "units", ""))
