# -*- coding: utf-8 -*-
# =====================================================================
# CELDA PARA LA NOTEBOOK Entrenamiento_Ext_Clip_U-Net_Comb_V2.1.ipynb
# (la Ext Clip U-Net Comb V2.1, el mejor modelo).
#
# Dónde pegarla: AL FINAL de la notebook ya corrida, que deja armados
# `modelo` (con los pesos de la mejor época), `checkpoint`,
# `val_dataset` (../dataset/val, 3 canales) y `device`.
#
# Qué hace: lo mismo que celda_campo_receptivo_efectivo.py con la Temp
# (gradiente de un píxel de salida respecto de la entrada, Luo et al.,
# 2016; 100 muestras con actividad x 20 píxeles; r50 y r90) y además el
# campo receptivo teórico de esta arquitectura (kernels 3, 5 y 3).
#
# Salida: imprime la tabla y guarda campo_efectivo_comb.npz. Mandame
# ese archivo y lo que imprime. Tarda unos minutos con GPU.
# =====================================================================
from fractions import Fraction

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

# --- 1) Parámetros de la medición ---------------------------------------
NOMBRE = "Ext Clip U-Net Comb V2.1"
N_MUESTRAS = 100      # muestras de validación con actividad
N_PIXELES = 20        # píxeles de salida por muestra
MARGEN = 100          # distancia mínima a los bordes del dominio
SEMILLA = 0
RUTA_SALIDA = "campo_efectivo_comb.npz"
KM_POR_PIXEL = 2.0
TAM = 297             # lado del dominio de entrada


# --- 2) Campo receptivo teórico (dependencias exactas) ------------------
def _conv(S, k, n):
    r = k // 2
    return {j for i in S for j in range(i - r, i + r + 1)
            if 0 <= j < n}


def _pool(S):                      # salida i depende de 2i y 2i+1
    return {2 * i + d for i in S for d in (0, 1)}


def _up(S, nlow):                  # bilineal x2, align_corners=True
    out = set()
    for i in S:
        s = Fraction(i * (nlow - 1), 2 * nlow - 1)
        f = s.numerator // s.denominator
        out.add(f)
        if s.denominator != 1 and f + 1 < nlow:
            out.add(f + 1)
    return out


def campo_teorico(kernels, tam=TAM):
    """Lado (en píxeles) del campo receptivo teórico de la U-Net con
    esos kernels por nivel (el cuello de botella usa el último).
    Se traza una dimensión (el campo es un cuadrado) y se toma el
    máximo sobre las posiciones del píxel de salida."""
    L = len(kernels)
    div = 2 ** L
    n1 = ((tam // div) + 1) * div            # tamaño con relleno
    pad_izq = (n1 - tam) // 2
    ns = [n1 // 2 ** i for i in range(L + 1)]
    ks = list(kernels) + [kernels[-1]]
    mejor = 0
    for p in range(MARGEN, tam - MARGEN):
        S = {p + pad_izq}
        skips = []
        for lev in range(L):
            S = _conv(_conv(S, ks[lev], ns[lev]), ks[lev], ns[lev])
            skips.append(S)
            S = _up(S, ns[lev + 1])
        S = _conv(_conv(S, ks[L], ns[L]), ks[L], ns[L])
        for lev in range(L - 1, -1, -1):
            S = _pool(S) | skips[lev]
            S = _conv(_conv(S, ks[lev], ns[lev]), ks[lev], ns[lev])
        mejor = max(mejor, max(S) - min(S) + 1)
    return mejor


# --- 3) Modelo y muestras -----------------------------------------------
modelo.load_state_dict(checkpoint["best_model_state_dict"])   # mejor época (13)
assert sum(p.numel() for p in modelo.parameters()) == 2_307_329
m = modelo
m.eval()
for p_ in m.parameters():
    p_.requires_grad_(False)            # solo hace falta d(salida)/d(x)
sigmoide_original = m.sigmoid
m.sigmoid = nn.Identity()               # medimos antes de la sigmoide

kernels = [b[0].kernel_size[0] for b in m.enc_blocks]
L_teo = campo_teorico(kernels)
print(f"{NOMBRE}: kernels {kernels}, campo teórico {L_teo}x{L_teo}")

ds = val_dataset
rng = np.random.default_rng(SEMILLA)
elegidas = []
for k in rng.permutation(len(ds)):
    _, y = ds[int(k)]
    if float(y.max()) > 0:               # hay actividad eléctrica
        elegidas.append(int(k))
    if len(elegidas) == N_MUESTRAS:
        break
print(f"Muestras con actividad elegidas: {len(elegidas)} "
      f"(de {len(ds)} de validación)")

# Píxeles de salida: posiciones en [MARGEN, TAM-MARGEN)
lado = TAM - 2 * MARGEN
pixeles = {k: [(MARGEN + int(i), MARGEN + int(j))
               for i, j in rng.integers(0, lado, size=(N_PIXELES, 2))]
           for k in elegidas}

# --- 4) Medición ----------------------------------------------------------
H = MARGEN                               # semiancho de la ventana
yy, xx = np.mgrid[-H:H + 1, -H:H + 1]
bins = np.floor(np.hypot(yy, xx)).astype(int)    # radio en píxeles
R_MAX = int(bins.max())


def acumulada(mapa):
    """Fracción acumulada de la influencia según el radio."""
    h = np.bincount(bins.ravel(), weights=mapa.ravel(),
                    minlength=R_MAX + 1)
    return np.cumsum(h) / h.sum()


def primer_radio(acum, frac):
    return int(np.argmax(acum >= frac))


suma = np.zeros((2 * H + 1, 2 * H + 1))
r50s, r90s, fuera = [], [], []
for k in tqdm(elegidas, desc=NOMBRE):
    X, _ = ds[k]
    x = X.unsqueeze(0).to(device).requires_grad_(True)
    salida = m(x)
    for (i, j) in pixeles[k]:
        g, = torch.autograd.grad(salida[0, 0, i, j], x,
                                 retain_graph=True)
        mapa = g.abs().sum(dim=1)[0].cpu().numpy().astype("float64")
        total = mapa.sum()
        if total <= 0:                   # ninguna ruta activa
            continue
        mapa /= total
        vent = mapa[i - H:i + H + 1, j - H:j + H + 1]
        fuera.append(1.0 - vent.sum())
        suma += vent
        ac = acumulada(vent)
        r50s.append(primer_radio(ac, 0.5))
        r90s.append(primer_radio(ac, 0.9))
m.sigmoid = sigmoide_original            # dejar `modelo` como estaba

n = len(r90s)
medio = suma / n
ac = acumulada(medio)
r50, r90 = primer_radio(ac, 0.5), primer_radio(ac, 0.9)
r50s, r90s = np.array(r50s), np.array(r90s)

# --- 5) Resultados --------------------------------------------------------
print("\n" + "=" * 70)
print(f"{NOMBRE}: {n} mapas")
print(f"  campo teórico      : {L_teo}x{L_teo} píxeles "
      f"(radio {L_teo // 2})")
print(f"  r50 (mapa medio)   : {r50} píxeles "
      f"({r50 * KM_POR_PIXEL:.0f} km)")
print(f"  r90 (mapa medio)   : {r90} píxeles "
      f"({r90 * KM_POR_PIXEL:.0f} km)")
print("Control: influencia fuera de la ventana de "
      f"{2 * H + 1}x{2 * H + 1} (debería ser ~0): "
      f"{np.mean(fuera):.2e}")
print(f"Dispersión entre mapas individuales: r90 media "
      f"{r90s.mean():.1f}, desvío {r90s.std():.1f}, mín {r90s.min()}, "
      f"máx {r90s.max()}; r50 media {r50s.mean():.1f}")
print("\nFilas de la tabla (LaTeX):")
print(f"  $r_{{50}}$ (píxeles) & {r50} \\\\")
print(f"  $r_{{90}}$ (píxeles) & {r90} \\\\")
print(f"  $r_{{90}}$ (km) & {r90 * KM_POR_PIXEL:.0f} \\\\")

np.savez(RUTA_SALIDA,
         nombres=np.array([NOMBRE]),
         medio=medio[None], acum=ac[None],
         r50=np.array([r50]), r90=np.array([r90]),
         L_teo=np.array([L_teo]),
         r90_ind=np.array([r90s.mean()]),
         km_por_pixel=KM_POR_PIXEL)
print(f"\nGuardado en {RUTA_SALIDA}")
