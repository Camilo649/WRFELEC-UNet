"""
Genera datos/limites_provincias.npz: límites provinciales (Natural
Earth, 1:10 m) proyectados a los píxeles de la grilla de 297x297 que
usan las figuras (columna = x, fila = y, fila 0 al sur).

La grilla se reconstruye con la proyección del dominio: cónica
conforme de Lambert, paralelo estándar en 38 °S, longitud de
referencia 63 °O, centro en 31,5 °S 62 °O, 2 km de espaciado, esfera
de 6.370 km (la del WRF). El centro de la grilla de 299 puntos es el
píxel 149; al recortar un píxel por borde queda en el 148.

Uso:  python3 figuras/preparar_limites.py [ne_10m_admin_1_..._lines.geojson]
(sin argumento lo baja de GitHub). Requiere pyproj.
"""
import json, os, sys, urllib.request
import numpy as np
from pyproj import CRS, Transformer

N, DX, CEN = 297, 2000.0, 148
URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector"
       "/master/geojson/ne_10m_admin_1_states_provinces_lines.geojson")
HERE = os.path.dirname(os.path.abspath(__file__))

wrf = CRS.from_proj4("+proj=lcc +lat_1=-38 +lat_2=-38 +lat_0=-38 "
                     "+lon_0=-63 +a=6370000 +b=6370000 +units=m")
tr = Transformer.from_crs("EPSG:4326", wrf, always_xy=True)
xc, yc = tr.transform(-62.0, -31.5)


def a_pixeles(lon, lat):
    x, y = tr.transform(np.asarray(lon), np.asarray(lat))
    return CEN + (x - xc) / DX, CEN + (y - yc) / DX


def grilla_lonlat():
    """lon, lat de los 297x297 píxeles (para comparar con XLONG/XLAT)."""
    j, i = np.meshgrid(np.arange(N), np.arange(N))
    inv = Transformer.from_crs(wrf, "EPSG:4326", always_xy=True)
    return inv.transform(xc + (j - CEN) * DX, yc + (i - CEN) * DX)


if __name__ == "__main__":
    ruta = sys.argv[1] if len(sys.argv) > 1 else None
    if ruta is None:
        ruta = "/tmp/ne_lines.geojson"
        urllib.request.urlretrieve(URL, ruta)
    g = json.load(open(ruta))
    partes = []
    for f in g["features"]:
        geo = f["geometry"]
        if geo is None:
            continue
        lineas = (geo["coordinates"] if geo["type"] == "MultiLineString"
                  else [geo["coordinates"]])
        for ln in lineas:
            ln = np.array(ln)
            col, fil = a_pixeles(ln[:, 0], ln[:, 1])
            ok = (col > -20) & (col < N + 20) & (fil > -20) & (fil < N + 20)
            if ok.any():
                # recorta a los vértices cercanos al dominio
                # incluye también el vértice vecino de cada tramo
                # interior, para que la línea siga hasta salir del
                # dominio en vez de cortarse en el último vértice
                # interior (el eje recorta lo que sobra)
                ok = ok | np.r_[ok[1:], False] | np.r_[False, ok[:-1]]
                idx = np.where(ok)[0]
                cortes = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
                for c in cortes:
                    partes.append(np.column_stack([col[c], fil[c]]))
                    partes.append(np.array([[np.nan, np.nan]]))
    L = np.vstack(partes)
    out = os.path.join(HERE, "datos", "limites_provincias.npz")
    np.savez_compressed(out, x=L[:, 0].astype(np.float32),
                        y=L[:, 1].astype(np.float32))
    lon, lat = grilla_lonlat()
    print("OK", out, len(L), "vértices")
    print("Esquinas (lon, lat) [SO, SE, NO, NE]:")
    for a, b in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        print(f"  {lon[a, b]:.3f}, {lat[a, b]:.3f}")
