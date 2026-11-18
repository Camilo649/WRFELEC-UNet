# -*- coding: utf-8 -*-
"""
Métricas del capítulo de Resultados, con las definiciones de la
Sección "Métricas de evaluación" de la monografía.

Todas se calculan solo sobre las columnas activas de cada paso de
tiempo (LIGHTDENS > 0 en el valor simulado o en el estimado) y no
se consideran los pasos de tiempo sin columnas activas.

Con estas funciones se reproducen las métricas que da la notebook
para el modelo final (MSE 186,07; SSIM 0,652; CSI 0,659; ...).
"""
import numpy as np
from scipy.ndimage import uniform_filter
from skimage.metrics import structural_similarity

UMBRALES = (1, 5, 15, 30)      # FSS: umbrales de LIGHTDENS
VECINDARIOS = (1, 3, 5, 15)    # FSS: lado del vecindario (columnas)


def metricas(Y, P, ssim=True, fss_cfg=((15, 3),)):
    """Y, P: arreglos (pasos, filas, columnas) con el valor simulado
    y el estimado, ya desnormalizados y redondeados.

    MSE, RMSE, MAE, SSIM, POD, FAR y CSI: promedio sobre los pasos
    de tiempo. R2, sesgo y FSS: globales."""
    mse, rmse, mae, ss, pod, far, csi = [], [], [], [], [], [], []
    n = sy = syy = sres = sdif = 0.0
    num = {c: 0.0 for c in fss_cfg}
    den = {c: 0.0 for c in fss_cfg}
    for y, p in zip(Y, P):
        y = y.astype(np.float64)
        p = p.astype(np.float64)
        act = (y > 0) | (p > 0)
        if not act.any():
            continue
        d = (y - p)[act]
        mse.append((d ** 2).mean())
        rmse.append(np.sqrt((d ** 2).mean()))
        mae.append(np.abs(d).mean())
        if ssim:
            _, mapa = structural_similarity(
                y, p, win_size=7, data_range=150, full=True)
            ss.append(mapa[act].mean())
        H = ((y > 0) & (p > 0)).sum()
        M = ((y > 0) & (p == 0)).sum()
        F = ((y == 0) & (p > 0)).sum()
        pod.append(H / (H + M) if H + M else np.nan)
        far.append(F / (H + F) if H + F else np.nan)
        csi.append(H / (H + M + F))
        n += act.sum()
        sy += y[act].sum()
        syy += (y[act] ** 2).sum()
        sres += (d ** 2).sum()
        sdif += (-d).sum()
        for (u, v) in fss_cfg:
            O = uniform_filter((y >= u).astype(np.float64), v,
                               mode="constant")
            Pf = uniform_filter((p >= u).astype(np.float64), v,
                                mode="constant")
            num[(u, v)] += ((O - Pf) ** 2)[act].sum()
            den[(u, v)] += (O ** 2 + Pf ** 2)[act].sum()
    ybar = sy / n
    out = dict(MSE=np.mean(mse), RMSE=np.mean(rmse), MAE=np.mean(mae),
               SSIM=np.mean(ss) if ssim else np.nan,
               R2=1 - sres / (syy - n * ybar ** 2), sesgo=sdif / n,
               POD=np.nanmean(pod), FAR=np.nanmean(far),
               CSI=np.mean(csi))
    for (u, v) in fss_cfg:
        out[f"FSS_u{u}_v{v}"] = 1 - num[(u, v)] / den[(u, v)]
    return {k: float(x) for k, x in out.items()}


def csi_por_paso(Y, P):
    """CSI de cada paso de tiempo (NaN si no hay columnas activas)."""
    out = np.full(len(Y), np.nan)
    for k, (y, p) in enumerate(zip(Y, P)):
        act = (y > 0) | (p > 0)
        if act.any():
            out[k] = ((y > 0) & (p > 0)).sum() / act.sum()
    return out
