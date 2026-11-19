# -*- coding: utf-8 -*-
"""Cifras de la Sección 7.1 y del borde sur (Capítulo 6): cociente estimado/simulado por fila desde el borde sur,
comparación paso a paso del MSE y del CSI de la Ext Clip U-Net Comb V2.1 con los otros dos modelos y
remuestreo con reposición de bloques de una hora (2.000 repeticiones) del MSE global.

Uso (desde la carpeta figuras/):
    python3 robustez_comb.py
"""
import numpy as np, datetime as dt
D=np.load('datos/resultados_validacion.npz')
Y=D['y_real']; K=['pred_ext_ultima','pred_comb_clip','pred_temp']
rowsY=np.zeros(297,np.int64)
for m in Y: rowsY+=m.sum(1,dtype=np.int64)
for k in K:
    r=np.zeros(297,np.int64)
    for m in D[k]: r+=m.sum(1,dtype=np.int64)
    rr=r/np.maximum(rowsY,1)
    print(k,'filas 1-12:',' '.join('%.2f'%x for x in rr[:12]))
    # bandas de 6 filas
    print('   bandas6:',' '.join('%.2f'%(r[a:a+6].sum()/rowsY[a:a+6].sum()) for a in range(0,48,6)), '| filas 7+ %.2f, 8+ %.2f'%(r[6:].sum()/rowsY[6:].sum(), r[7:].sum()/rowsY[7:].sum()))
print('act por fila sim (filas 1-12, % del total):',' '.join('%.2f'%(100*x/rowsY.sum()) for x in rowsY[:12]))
# comparaciones por paso de tiempo
def per(P):
    mse=[];csi=[]
    for y,p in zip(Y,P):
        a=(y>0)|(p>0)
        if not a.any(): mse.append(np.nan); csi.append(np.nan); continue
        d=(y[a].astype(float)-p[a])
        mse.append((d*d).mean()); csi.append(((y>0)&(p>0)).sum()/a.sum())
    return np.array(mse),np.array(csi)
M={k:per(D[k]) for k in K}
A=[str(a) for a in D['archivos']]
T=[dt.datetime.strptime(a[11:30],"%Y-%m-%d_%H-%M-%S") for a in A]
cortes=[0]+[i for i in range(1,len(T)) if (T[i]-T[i-1]).total_seconds()>120]+[len(T)]
tot=lambda P:np.array([m.sum(dtype=np.int64) for m in P],float)
sY=tot(Y); S={k:tot(D[k]) for k in K}
for o in ['pred_ext_ultima','pred_temp']:
    c,e=M['pred_comb_clip'],M[o]; ok=~np.isnan(c[0])&~np.isnan(e[0])
    print('comb vs',o,'pasos con MSE menor %.0f%%, CSI mayor %.0f%%'%(100*(c[0][ok]<e[0][ok]).mean(),100*(c[1][ok]>e[1][ok]).mean()),
          '| |error actividad total| menor %.0f%%'%(100*(np.abs(S['pred_comb_clip']-sY)<np.abs(S[o]-sY)).mean()))
# bootstrap por bloques de 1 h (30 pasos) del MSE global
rng=np.random.default_rng(0)
blocks=[(a,min(a+30,b)) for a,b in zip(cortes[:-1],cortes[1:]) for a in range(a,b,30)]
def sse(P):
    s=np.zeros(len(Y));n=np.zeros(len(Y))
    for i,(y,p) in enumerate(zip(Y,P)):
        a=(y>0)|(p>0); d=y[a].astype(float)-p[a]; s[i]=(d*d).sum(); n[i]=a.sum()
    return s,n
SS={k:sse(D[k]) for k in K}
bs=[]
for _ in range(2000):
    idx=rng.integers(0,len(blocks),len(blocks))
    sel=np.concatenate([np.arange(*blocks[j]) for j in idx])
    v={k:SS[k][0][sel].sum()/SS[k][1][sel].sum() for k in K}
    bs.append((v['pred_comb_clip']<v['pred_ext_ultima'], v['pred_comb_clip']<v['pred_temp'], v['pred_comb_clip']/v['pred_ext_ultima']))
bs=np.array(bs,float)
print('bootstrap bloques 1h: MSE comb<ext %.1f%%, comb<temp %.1f%%, cociente comb/ext IC95 %.2f-%.2f'%(100*bs[:,0].mean(),100*bs[:,1].mean(),*np.percentile(bs[:,2],[2.5,97.5])))
