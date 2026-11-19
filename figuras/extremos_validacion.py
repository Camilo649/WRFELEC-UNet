# -*- coding: utf-8 -*-
"""
Valores extremos de LIGHTDENS en el conjunto de validación: cantidad
de columnas (sumadas sobre las 2.271 muestras) en que el valor
simulado y el estimado por cada modelo alcanzan o superan 100 y
llegan a 150 (el valor al que se restringió la variable objetivo),
y valor máximo.

Uso (desde la carpeta de la monografía):
    python3 figuras/extremos_validacion.py
Lee figuras/datos/resultados_validacion.npz.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(HERE, "datos", "resultados_validacion.npz"))
NOMBRES = {"y_real": "WRF-ELEC",
           "pred_comb_clip": "Ext Clip U-Net Comb V2.1",
           "pred_ext_ultima": "Ext Clip U-Net FACL",
           "pred_temp": "Temp Ext Clip U-Net FACL"}
print(f"{'':28s}{'>= 100':>10s}{'= 150':>10s}{'máximo':>8s}")
for k, n in NOMBRES.items():
    M = D[k]
    print(f"{n:28s}{int((M >= 100).sum()):>10,d}"
          f"{int((M == 150).sum()):>10,d}{int(M.max()):>8d}")
