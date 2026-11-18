# -*- coding: utf-8 -*-
"""
Serie de descargas del LMA cada 2 minutos (datos/lma_serie_2min.csv).

Cuenta, para cada día pedido, los flashes del LMA por intervalo de
2 minutos (a partir de la columna `start`, segundos del día UTC) a
menos de 100 km del centro de la red (el CSV ya viene recortado a
ese radio). Los intervalos sin descargas entre el primero y el
último registro del día se completan con cero.

Uso:  python3 figuras/lma_serie_2min.py ruta/lma_claude.csv \
          2018-12-14 2019-01-25 2019-02-09
"""
import sys
import numpy as np
import pandas as pd


def serie(df, dia):
    d = df[df.fecha == dia]
    b = (d.start // 120).astype(int)
    c = b.value_counts().sort_index()
    idx = np.arange(c.index.min(), c.index.max() + 1)
    c = c.reindex(idx, fill_value=0)
    return pd.DataFrame({"fecha": dia, "hora_utc": idx * 120 / 3600,
                         "descargas": c.values})


if __name__ == "__main__":
    ruta, dias = sys.argv[1], sys.argv[2:]
    df = pd.read_csv(ruta, usecols=["start", "fecha"])
    out = pd.concat([serie(df, d) for d in dias])
    out.to_csv("figuras/datos/lma_serie_2min.csv", index=False)
    print(out.groupby("fecha").descargas.agg(["count", "sum", "max"]))
