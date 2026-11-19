# -*- coding: utf-8 -*-
"""Cifras del Capítulo 6: valores extremos columna a columna, escala normalizada, series por tormenta, CSI por paso de tiempo, borde sur y paso de tiempo de la figura de comparación.

Uso (desde la carpeta figuras/):
    python3 cifras_resultados.py
Lee figuras/datos/resultados_validacion.npz (con la clave pred_comb_clip, que agrega
figuras/celda_resultados_comb.py).
"""
import os, numpy as np, datetime as dt
D=np.load('datos/resultados_validacion.npz')
Y=D['y_real']
K=['pred_ext_ultima','pred_comb_clip','pred_temp']
A=[str(a) for a in D['archivos']]
T=[dt.datetime.strptime(a[11:30],"%Y-%m-%d_%H-%M-%S") for a in A]
cortes=[0]+[k for k in range(1,len(T)) if (T[k]-T[k-1]).total_seconds()>120]+[len(T)]
torm=list(zip(cortes[:-1],cortes[1:]))
lo,hi_=np.log(0.1),np.log(150.1)
nz=lambda M:(np.log(M.astype(np.float64)+0.1)-lo)/(hi_-lo)
for k in K:
    P=D[k]
    st=dict(n_hi=0,act_hi=0,sum_hi=0,ge100_hi=0,ph=0,ph_hi=0,dif=0,dif_hi=0,nb=0,yb=0,pb=0,ae=0,ae_hi=0,en=0,en_hi=0,
            ex=0,var0=0,ind=0)
    for t in range(len(Y)):
        y=Y[t].astype(np.int64); p=P[t].astype(np.int64)
        hi=y>=100; ph=p>=100
        st['n_hi']+=hi.sum(); st['act_hi']+=(p[hi]>0).sum(); st['sum_hi']+=p[hi].sum(); st['ge100_hi']+=(p[hi]>=100).sum()
        st['ph']+=ph.sum(); st['ph_hi']+=(ph&hi).sum()
        st['dif']+=(y-p).sum(); st['dif_hi']+=(y-p)[hi].sum()
        b=(y>0)&(p>0); st['nb']+=nz(p[b]).sum(); st['yb']+=nz(y[b]).sum(); st['pb']+=p[b].sum(); st['sb']=st.get('sb',0)+y[b].sum()
        ae=np.abs(y-p); st['ae']+=ae.sum(); st['ae_hi']+=ae[hi].sum()
        u=(y>0)|(p>0); en=np.abs(nz(y[u])-nz(p[u])); st['en']+=en.sum(); st['en_hi']+=en[y[u]>=100].sum()
        if not u.any(): st['ex']+=1; continue
        if np.var(y[u].astype(float))==0: st['var0']+=1
        if not ((y>=15).any() or (p>=15).any()): st['ind']+=1
    s=st
    print(k,'act en sim>=100 %.1f%% prom %.1f est>=100 %.1f%% | est>=100 con sim>=100 %.1f%% | aporte dif %.0f%% | ambos norm %.0f%% desc %.0f%% | |err| desc hi %.1f%% norm hi %.1f%% | excl %d var0 %d fssind %d'%(
      100*s['act_hi']/s['n_hi'], s['sum_hi']/s['n_hi'], 100*s['ge100_hi']/s['n_hi'], 100*s['ph_hi']/s['ph'], 100*s['dif_hi']/s['dif'],
      100*s['nb']/s['yb'], 100*s['pb']/s['sb'], 100*s['ae_hi']/s['ae'], 100*s['en_hi']/s['en'], s['ex'], s['var0'], s['ind']),flush=True)
    del P
tot=lambda M:np.array([m.sum(dtype=np.int64) for m in M],float)
sY=tot(Y); S={k:tot(D[k]) for k in K+['pred_ext_epoca8']}
print('--- series (est/sim, corr) por tormenta y total')
for k,s in S.items():
    print(k,' '.join('%.2f/%.2f'%(s[a:b].sum()/sY[a:b].sum(),np.corrcoef(sY[a:b],s[a:b])[0,1]) for a,b in torm+[(0,len(Y))]))
for a_,b_ in [('pred_ext_ultima','pred_comb_clip'),('pred_comb_clip','pred_temp'),('pred_ext_ultima','pred_temp')]:
    print(a_,b_,'corr', ' '.join('%.2f'%np.corrcoef(S[a_][a:b],S[b_][a:b])[0,1] for a,b in torm+[(0,len(Y))]))
print('--- CSI')
def mm(x,n=15):
    ok=~np.isnan(x); num=np.convolve(np.where(ok,x,0),np.ones(n),'same'); den=np.convolve(ok.astype(float),np.ones(n),'same')
    return np.where(den>0,num/np.maximum(den,1),np.nan)
for k in K:
    P=D[k]; c=np.full(len(Y),np.nan)
    for t in range(len(Y)):
        a=(Y[t]>0)|(P[t]>0)
        if a.any(): c[t]=((Y[t]>0)&(P[t]>0)).sum()/a.sum()
    sm=np.concatenate([mm(c[a:b]) for a,b in torm]); v=sm[~np.isnan(sm)]
    print(k,'suav 0.6-0.8 %.0f%% | <0.2 %.1f%% | medias %s'%(100*((v>=0.6)&(v<=0.8)).mean(),100*(v<0.2).mean(),' '.join('%.2f'%np.nanmean(c[a:b]) for a,b in torm)))
    del P
print('--- borde')
rowsY=np.zeros(297,np.int64); colsY=np.zeros(297,np.int64)
for m in Y: rowsY+=m.sum(1,dtype=np.int64); colsY+=m.sum(0,dtype=np.int64)
n150=sum(int((m==150).sum()) for m in Y); n150r=sum(int((m[:6]==150).sum()) for m in Y)
print('6 filas: %.1f%% cols, %.1f%% act, %.1f%% de 150'%(100*6/297,100*rowsY[:6].sum()/rowsY.sum(),100*n150r/n150))
for k in K:
    P=D[k]; r=np.zeros(297,np.int64); cl=np.zeros(297,np.int64)
    for m in P: r+=m.sum(1,dtype=np.int64); cl+=m.sum(0,dtype=np.int64)
    print(k,'6 filas %.2f resto %.2f | filas 7-12 %s | este %s | total %.3f sin franja %.3f'%(r[:6].sum()/rowsY[:6].sum(), r[6:].sum()/rowsY[6:].sum(),
      ' '.join('%.2f'%(r[i]/rowsY[i]) for i in range(6,12)), ' '.join('%.2f'%(cl[-i]/colsY[-i]) for i in range(1,7)), r.sum()/rowsY.sum(), r[6:].sum()/rowsY[6:].sum()))
    del P
i=A.index('wrfout_d01_2018-12-14_01-00-00.npz')
for k in ['y_real']+K:
    m=D[k][i]; print('comp',k,'cols %d total %d max %d'%((m>0).sum(),m.sum(dtype=np.int64),m.max()))
