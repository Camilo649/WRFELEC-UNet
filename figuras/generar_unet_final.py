# -*- coding: utf-8 -*-
"""
Esquema de la U-Net del mejor modelo (Ext Clip U-Net Comb V2.1) con mapas reales.

Uso (desde la carpeta de la monografía):
    python3 figuras/generar_unet_final.py

Dibuja la arquitectura del mejor modelo (3 canales de entrada,
3 niveles, 32 filtros base, kernels 3, 5 y 3; ensayo 92 de la Fase 2,
época 13) con las variables de un paso de tiempo real de validación
(14/12/2018, 01:00 UTC; los tres canales en t de
figuras/datos/memo_trial100_instante.npz) y la estimación del modelo
(figuras/datos/resultados_validacion.npz, clave pred_comb_clip);
superpone los límites provinciales (datos/limites_provincias.npz).
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import unet_diagram as ud  # noqa: E402
from generar_figuras_instante import (  # noqa: E402
    SEQ, SEQ_FRIO, SEQ_RAYOS, DIV)

D = np.load(os.path.join(HERE, "datos", "memo_trial100_instante.npz"))
R = np.load(os.path.join(HERE, "datos", "resultados_validacion.npz"))
_k = [str(a) for a in R["archivos"]].index(str(D["archivo"]))
LIM = np.load(os.path.join(HERE, "datos", "limites_provincias.npz"))

# Los límites provinciales ya están en píxeles de la grilla: se los
# entrega al esquema como "contornos" con un bbox en píxeles.
ud._contornos = lambda *a, **k: [(LIM["x"].astype(float),
                                  LIM["y"].astype(float))]

pred = R["pred_comb_clip"][_k].astype(float)
ud.draw_unet(
    in_channels=3, n_niveles=3, n_filtros_base=32,
    kernel_sizes=[3, 5, 3],
    X_sample=D["X"][:3],
    y_pred_sample=pred,
    var_names=["MDBZ", "HIELO", "CTT"],
    var_cmaps=[SEQ, SEQ, SEQ_FRIO],
    target_cmap=SEQ_RAYOS,
    lon_min=-0.5, lon_max=296.5, lat_min=-0.5, lat_max=296.5,
    out_path=os.path.join(HERE, "..", "img", "redes_unet.pdf"))
