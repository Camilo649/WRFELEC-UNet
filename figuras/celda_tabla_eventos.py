# -*- coding: utf-8 -*-
"""
Tabla de eventos (Tabla 4.x). Pegar en una celda de Datasets.ipynb,
donde ya existe `eventos` (celda "Corridas continuas"); imprime las
filas de la tabla en LaTeX para copiarlas en el chat.
"""
import os
import re
from datetime import datetime

PAT = re.compile(r"wrfout_d01_(\d{4}-\d{2}-\d{2})_(\d{2})")


def dia(ruta):
    m = PAT.search(os.path.basename(str(ruta)))
    return datetime.strptime(m.group(1), "%Y-%m-%d")


def fmt_dias(ds):
    ds = sorted(ds)
    if len(ds) == 1:
        return ds[0].strftime("%d/%m/%Y")
    if ds[0].month == ds[-1].month and ds[0].year == ds[-1].year:
        return "%s a %s" % (ds[0].strftime("%d"),
                            ds[-1].strftime("%d/%m/%Y"))
    return "%s a %s" % (ds[0].strftime("%d/%m/%Y"),
                        ds[-1].strftime("%d/%m/%Y"))


filas = []
for k, fs in eventos.items():
    ds = sorted({dia(f) for f in fs})
    filas.append((ds[0], fmt_dias(ds), len(ds), len(fs)))
filas.sort()
print(len(filas), "eventos;", sum(f[3] for f in filas), "salidas")
for i, (_, txt, nd, ns) in enumerate(filas, 1):
    print("    %d & %s & %d & %s \\\\" % (
        i, txt, nd, f"{ns:,}".replace(",", "\\,")))
