# -*- coding: utf-8 -*-
"""
Esquema de la U-Net final (Fig. 4.1) con mapas reales.

Uso (desde la carpeta de la monografía):
    python3 figuras/generar_unet_final.py

Dibuja la arquitectura del modelo final (9 canales de entrada,
3 niveles, 32 filtros base, kernels 5, 5 y 5; ensayo 100) con las
variables de un instante real de validación
(figuras/datos/temp_trial100_instante.npz, 14/12/2018, 01:00 UTC)
y la estimación del modelo; superpone
los límites provinciales (datos/limites_provincias.npz).
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import unet_diagram as ud  # noqa: E402
from generar_figuras_instante import (  # noqa: E402
    SEQ, SEQ_FRIO, SEQ_RAYOS, DIV)

D = np.load(os.path.join(HERE, "datos", "temp_trial100_instante.npz"))
LIM = np.load(os.path.join(HERE, "datos", "limites_provincias.npz"))

# Los límites provinciales ya están en píxeles de la grilla: se los
# entrega al esquema como "contornos" con un bbox en píxeles.
ud._contornos = lambda *a, **k: [(LIM["x"].astype(float),
                                  LIM["y"].astype(float))]

pred = np.round(D["pred"])
pred[pred < 0.5] = 0.0
ud.draw_unet(
    in_channels=9, n_niveles=3, n_filtros_base=32,
    kernel_sizes=[5, 5, 5],
    X_sample=D["X"],
    y_pred_sample=pred,
    var_names=["MDBZ", "HIELO", "CTT"],
    var_cmaps=[SEQ, SEQ, SEQ_FRIO],
    diff_cmap=DIV,
    # valor normalizado de una diferencia nula (MDBZ, HIELO, CTT)
    diff_cero=[0.456, 0.5185, 0.5],
    target_cmap=SEQ_RAYOS,
    lon_min=-0.5, lon_max=296.5, lat_min=-0.5, lat_max=296.5,
    out_path=os.path.join(HERE, "..", "img", "redes_unet.pdf"))
