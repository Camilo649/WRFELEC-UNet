# -*- coding: utf-8 -*-
"""
Máscara del círculo de 100 km del LMA sobre los mapas de 297x297,
reconstruida a partir de la proyección del dominio (Tabla de
configuración del WRF-ELEC): Lambert conforme, paralelo estándar
38°S, longitud de referencia 63°O, centro 31,5°S 62°O, 299x299
puntos de masa cada 2 km, esfera de 6370 km (la del WRF).
Escribe datos/mascara_radio_lma_297.npy (fila 0 = sur).
"""
import os
import numpy as np

R = 6370000.0
PHI1, LAM0 = np.radians(-38.0), np.radians(-63.0)
PHIC, LAMC = np.radians(-31.5), np.radians(-62.0)
NX, DX = 299, 2000.0
LAT0, LON0, RADIO_KM = -31.7, -64.1, 100.0

n = np.sin(PHI1)
F = np.cos(PHI1) * np.tan(np.pi / 4 + PHI1 / 2) ** n / n


def rho(phi):
    return R * F / np.tan(np.pi / 4 + phi / 2) ** n


def directa(phi, lam):
    r = rho(phi)
    return r * np.sin(n * (lam - LAM0)), -r * np.cos(n * (lam - LAM0))


def inversa(x, y):
    r = np.sign(n) * np.hypot(x, y)
    th = np.arctan2(np.sign(n) * x, -np.sign(n) * y)
    phi = 2 * np.arctan((R * F / r) ** (1 / n)) - np.pi / 2
    return phi, LAM0 + th / n


xc, yc = directa(PHIC, LAMC)
k = (np.arange(NX) - (NX - 1) / 2) * DX
X, Y = np.meshgrid(xc + k, yc + k)          # fila 0 = sur
phi, lam = inversa(X, Y)
lat, lon = np.degrees(phi), np.degrees(lam)
p1, p2 = np.radians(lat), np.radians(LAT0)
a = (np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2)
     * np.sin(np.radians(lon - LON0) / 2) ** 2)
dist = 2 * 6371.0 * np.arcsin(np.sqrt(a))
dentro = dist <= RADIO_KM
m297 = dentro[1:-1, 1:-1]
if __name__ == "__main__":
    print("esquinas lat/lon:", lat[0, 0], lon[0, 0], lat[-1, -1], lon[-1, -1])
    print("centro:", lat[149, 149], lon[149, 149])
    print("columnas dentro (299):", dentro.sum(), "| 297:", m297.sum())
    ys, xs = np.nonzero(m297)
    print("centro del círculo (fila, col):", ys.mean(), xs.mean())
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "datos", "mascara_radio_lma_297.npy"), m297)
