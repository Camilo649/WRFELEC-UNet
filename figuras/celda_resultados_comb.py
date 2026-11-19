# -*- coding: utf-8 -*-
# =====================================================================
# CELDA DE RESULTADOS (Capítulo 6) para la Ext Clip U-Net Comb V2.1.
#
# Dónde pegarla: AL FINAL de Entrenamiento_Ext_Clip_U-Net_Comb_V2.1.ipynb,
# con la notebook ya corrida (necesita `modelo` con los pesos de la
# mejor época cargados, `val_dataset`, `checkpoint` y `device`).
#
# Qué hace:
#   1. Corre el modelo sobre las 2.271 muestras de validación y guarda
#      el valor simulado y la estimación (desnormalizados y
#      redondeados, como en las métricas del Capítulo 6) en
#      resultados_validacion_comb.npz, con los nombres de archivo para
#      alinearlas con las de los otros dos modelos.
#   2. Mide el tiempo de inferencia (GPU lote 1, GPU lote 16, CPU con
#      un hilo) y la memoria de GPU (inferencia lote 1, paso de
#      entrenamiento lote 8), con el mismo procedimiento que se usó
#      para los otros dos modelos (celda_tiempo_inferencia.py).
#   3. Imprime el tiempo de entrenamiento según el checkpoint.
#
# Tarda unos minutos con GPU. Mandame resultados_validacion_comb.npz y
# todo lo que imprime. Si podés, corrélo sin otros entrenamientos en
# la GPU.
# =====================================================================
import copy
import time

import numpy as np
import torch
from tqdm.auto import tqdm

RUTA_SALIDA = "resultados_validacion_comb.npz"
LOTE = 16
TAM = 297
N_CALENT, N_MED, N_CPU = 20, 200, 20
LOTE_GRANDE, LOTE_ENTR = 16, 8
N_MUESTRAS = 32
ES_CUDA = device.type == "cuda"

n_par = sum(p.numel() for p in modelo.parameters())
assert n_par == 2_307_329, f"`modelo` no es la Comb V2.1 ({n_par:,})"
modelo.load_state_dict(checkpoint["best_model_state_dict"])   # mejor época
modelo.eval()
ds = getattr(val_dataset, "dataset", val_dataset)


# --- 1) Estimaciones sobre todo el conjunto de validación ------------
def _a_entero(t):
    """Desnormaliza un mapa y lo redondea, como en las métricas."""
    v = np.round(ds.desnormalizar_y(t).numpy())
    assert v.min() >= 0 and v.max() <= 255, (v.min(), v.max())
    return v.astype(np.uint8)


orden = sorted(range(len(ds)), key=lambda i: ds.files[i].name)
nombres = [ds.files[i].name for i in orden]
real = np.zeros((len(orden), TAM, TAM), dtype=np.uint8)
pred = np.zeros((len(orden), TAM, TAM), dtype=np.uint8)
with torch.no_grad():
    for a in tqdm(range(0, len(orden), LOTE), desc="Ext Clip Comb V2.1"):
        ids = orden[a:a + LOTE]
        muestras = [ds[i] for i in ids]
        X = torch.stack([m[0] for m in muestras]).to(device)
        p = modelo(X).cpu().squeeze(1).clamp(0, 1)
        for k, (_, y) in enumerate(muestras):
            real[a + k] = _a_entero(y.squeeze())
            pred[a + k] = _a_entero(p[k])

csis = []
for r, q in zip(real, pred):
    act = (r > 0) | (q > 0)
    if act.any():
        csis.append(((r > 0) & (q > 0)).sum() / act.sum())
print(f"Muestras: {len(nombres)} (esperado: 2271) | CSI medio "
      f"{np.mean(csis):.3f} (la notebook dio 0,670) | máximo estimado "
      f"{pred.max()}")
np.savez_compressed(RUTA_SALIDA, archivos=np.array(nombres),
                    y_real=real, pred_comb_clip=pred)
print(f"Guardado {RUTA_SALIDA} ({os.path.getsize(RUTA_SALIDA)/1e6:.1f} MB)")
del real, pred


# --- 2) Tiempo de inferencia y memoria ------------------------------
def _sync():
    if ES_CUDA:
        torch.cuda.synchronize()


def _ms(ts):
    ts = np.array(ts) * 1e3
    return (f"{ts.mean():8.2f} ms  (desvío {ts.std():.2f}, "
            f"mediana {np.median(ts):.2f}, mín {ts.min():.2f})")


print("\n" + "=" * 68)
print("PyTorch", torch.__version__, "| dispositivo:", device)
if ES_CUDA:
    print("GPU:", torch.cuda.get_device_name(0))
print(f"Ext Clip U-Net Comb V2.1: {n_par:,} parámetros ({n_par*4/1e6:.2f} MB)")
rng = np.random.default_rng(0)
ids = rng.permutation(len(ds))[:N_MUESTRAS]
xs = [ds[int(i)][0].unsqueeze(0).to(device) for i in ids]

with torch.no_grad():                       # GPU, lote de 1
    for k in range(N_CALENT):
        modelo(xs[k % len(xs)])
    _sync()
    ts = []
    for k in range(N_MED):
        x = xs[k % len(xs)]
        _sync(); t0 = time.perf_counter()
        modelo(x)
        _sync(); ts.append(time.perf_counter() - t0)
print(f"  {device.type.upper()}, lote de 1        : {_ms(ts)}")

xb = torch.cat(xs[:LOTE_GRANDE])            # GPU, lote de 16
with torch.no_grad():
    for _ in range(5):
        modelo(xb)
    _sync()
    ts = []
    for _ in range(30):
        _sync(); t0 = time.perf_counter()
        modelo(xb)
        _sync(); ts.append((time.perf_counter() - t0) / LOTE_GRANDE)
print(f"  {device.type.upper()}, lote de {LOTE_GRANDE} (c/u) : {_ms(ts)}")
del xb

hilos = torch.get_num_threads()             # CPU, lote de 1, un hilo
torch.set_num_threads(1)
mc = copy.deepcopy(modelo).to("cpu").eval()
xc = [x.cpu() for x in xs[:4]]
with torch.no_grad():
    mc(xc[0])
    ts = []
    for k in range(N_CPU):
        t0 = time.perf_counter()
        mc(xc[k % len(xc)])
        ts.append(time.perf_counter() - t0)
print(f"  CPU ({torch.get_num_threads()} hilo), lote de 1  : {_ms(ts)}")
torch.set_num_threads(hilos)
del mc, xc

if ES_CUDA:                                 # memoria de GPU
    torch.cuda.empty_cache(); _sync()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.no_grad():
        modelo(xs[0])
    _sync()
    print(f"  Memoria extra en inferencia (lote 1)      : "
          f"{(torch.cuda.max_memory_allocated() - base) / 1e6:8.1f} MB")
    banderas = [p.requires_grad for p in modelo.parameters()]
    for p in modelo.parameters():
        p.requires_grad_(True)
    modelo.train()
    xb = torch.cat(xs[:LOTE_ENTR])
    torch.cuda.empty_cache(); _sync()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = modelo(xb)
    out.mean().backward()
    _sync()
    print(f"  Memoria extra en entrenamiento (lote {LOTE_ENTR})  : "
          f"{(torch.cuda.max_memory_allocated() - base) / 1e6:8.1f} MB")
    modelo.zero_grad(set_to_none=True)
    modelo.eval()
    for p, b in zip(modelo.parameters(), banderas):
        p.requires_grad_(b)
    del out, xb
    torch.cuda.empty_cache()
else:
    print("  (sin CUDA: no se mide la memoria de GPU)")

# --- 3) Tiempo de entrenamiento (Clementina) -------------------------
seg = float(np.sum(checkpoint["historial_tiempo_epoch"]))
n_ep = checkpoint["epochs_entrenadas"]
print(f"\nEntrenamiento: {n_ep} épocas, {seg/3600:.2f} h en total "
      f"({int(seg//3600)} h {int(seg%3600//60)} min), "
      f"{seg/n_ep/60:.2f} min por época")
print("\nListo: mandame resultados_validacion_comb.npz y toda esta salida.")
