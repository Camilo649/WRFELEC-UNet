# -*- coding: utf-8 -*-
"""
Lee los atributos globales de un wrfout (guardan las opciones del
namelist con las que se corrió la simulación) para completar la
Tabla de configuración. Pegar en una celda de Datasets.ipynb, donde
ya existe `eventos`; imprime el resultado para copiarlo en el chat.
"""
import numpy as np
from netCDF4 import Dataset

CLAVES = [
    "TITLE", "GRIDTYPE", "DYN_OPT", "MAP_PROJ", "CEN_LAT", "CEN_LON",
    "MOAD_CEN_LAT", "TRUELAT1", "TRUELAT2", "STAND_LON", "POLE_LAT",
    "POLE_LON", "DX", "DY", "DT",
    "WEST-EAST_GRID_DIMENSION", "SOUTH-NORTH_GRID_DIMENSION",
    "BOTTOM-TOP_GRID_DIMENSION", "MP_PHYSICS", "RA_LW_PHYSICS",
    "RA_SW_PHYSICS", "SF_SFCLAY_PHYSICS", "SF_SURFACE_PHYSICS",
    "BL_PBL_PHYSICS", "CU_PHYSICS", "SF_URBAN_PHYSICS", "DIFF_OPT",
    "KM_OPT", "DAMPING_OPT", "ADV_OPT", "MOIST_ADV_OPT",
    "SCALAR_ADV_OPT", "W_DAMPING", "SIMULATION_START_DATE",
    "NUM_LAND_CAT", "ISWATER",
]


def leer_atributos(ruta):
    with Dataset(ruta) as ds:
        for k in CLAVES:
            if k in ds.ncattrs():
                v = getattr(ds, k)
                print(f"{k:28s} {np.asarray(v).tolist()}")
            else:
                print(f"{k:28s} (no está)")
        otros = [k for k in ds.ncattrs()
                 if any(s in k.upper() for s in
                        ("NSSL", "ELEC", "LIGHT", "RADT", "BLDT",
                         "SUITE"))]
        print("\nOtros atributos relacionados:", otros)
        for k in otros:
            print(f"{k:28s} {np.asarray(getattr(ds, k)).tolist()}")


# primera salida del primer evento
leer_atributos(eventos[sorted(eventos)[0]][0])
