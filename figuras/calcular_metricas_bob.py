import json, numpy as np, sys, warnings
warnings.filterwarnings('ignore')
sys.argv=['x']
# Uso: desde figuras/ (lee datos/resultados_validacion.npz)
from generar_figuras_resultados import D, REAL, tormentas
from metricas_resultados import metricas
Z=np.zeros_like(REAL)
out={'total':metricas(REAL,Z,fss_cfg=((15,3),(30,3)))}
print('total',out['total'],flush=True)
for i,(a,b) in enumerate(tormentas()):
    out[f't{i}']=metricas(REAL[a:b],Z[a:b],fss_cfg=((15,3),(30,3)))
    print(i,(a,b),out[f't{i}'],flush=True)
json.dump(out,open('datos/tabla_metricas_bob.json','w'),indent=1)
