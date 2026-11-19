# -*- coding: utf-8 -*-
"""MAE, RMSE y CSI entre las estimaciones de cada par de modelos y correlación entre sus errores.

Uso (desde la carpeta figuras/):
    python3 comparacion_entre_modelos.py
Lee figuras/datos/resultados_validacion.npz (con la clave pred_comb_clip, que agrega
figuras/celda_resultados_comb.py).
"""
import os, numpy as np, itertools
from metricas_resultados import metricas
D=np.load('datos/resultados_validacion.npz')
Y=D['y_real']; P={k:D[k] for k in ['pred_ext_ultima','pred_comb_clip','pred_temp']}
for a,b in itertools.combinations(P,2):
    m=metricas(P[a],P[b],ssim=False,fss_cfg=())
    n=sa=sb=saa=sbb=sab=0.0
    for k in range(len(Y)):
        y=Y[k].astype(np.int64); e=P[a][k].astype(np.int64); t=P[b][k].astype(np.int64)
        u=(y>0)|(e>0)|(t>0); x1=(e-y)[u]; x2=(t-y)[u]
        n+=x1.size; sa+=x1.sum(); sb+=x2.sum(); saa+=(x1*x1).sum(); sbb+=(x2*x2).sum(); sab+=(x1*x2).sum()
    r=(sab/n-sa*sb/n**2)/np.sqrt((saa/n-(sa/n)**2)*(sbb/n-(sb/n)**2))
    print(a,b,'MAE %.2f RMSE %.2f CSI %.2f corr_err %.3f n=%d'%(m['MAE'],m['RMSE'],m['CSI'],r,n),flush=True)
