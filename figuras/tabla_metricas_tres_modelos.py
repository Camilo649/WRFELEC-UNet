# -*- coding: utf-8 -*-
"""Métricas de la Tabla 6.1 (conjunto completo y por tormenta) para los tres modelos del Capítulo 6.

Uso (desde la carpeta figuras/):
    python3 tabla_metricas_tres_modelos.py
Lee figuras/datos/resultados_validacion.npz (con la clave pred_comb_clip, que agrega
figuras/celda_resultados_comb.py).
"""
import os, json, numpy as np, sys
from generar_figuras_resultados import D, REAL, tormentas
from metricas_resultados import metricas
claves=['pred_ext_ultima','pred_comb_clip','pred_temp']
out={}
for c in claves:
    P=D[c]
    out[c]={'total':metricas(REAL,P,fss_cfg=((15,3),(30,3)))}
    for i,(a,b) in enumerate(tormentas()):
        out[c][f't{i}']=metricas(REAL[a:b],P[a:b],fss_cfg=((15,3),(30,3)))
    print(c, out[c]['total'], flush=True)
json.dump(out,open('datos/tabla_metricas3.json','w'),indent=1)
