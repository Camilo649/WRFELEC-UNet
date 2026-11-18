# -*- coding: utf-8 -*-
# =====================================================================
# CELDA 2 DE RESULTADOS (Capítulo 6) — tiempo de inferencia y memoria.
#
# Dónde pegarla: AL FINAL de Entrenamiento_Temp_Ext_Clip_U-Net_FACL.ipynb,
# después de la celda "CHECKPOINT DE LA ETAPA B OFICIAL — ensayo 100,
# mejor época (best.pt)" (necesita `modelo`, `val_dataset`,
# `UNet_WRF`, `TormentaDataset` y `device`). No depende de la celda 1.
#
# Qué mide, para los dos últimos modelos (Ext Clip U-Net FACL y Temp
# Ext Clip U-Net FACL):
#   1. Tiempo de inferencia por paso de tiempo en la GPU: solo el pase
#      hacia adelante, con la entrada ya en la GPU, lote de 1, 200
#      repeticiones después de 20 de calentamiento.
#   2. Lo mismo con lote de 16 (tiempo por paso de tiempo al estimar
#      varios a la vez).
#   3. Tiempo de inferencia en CPU (lote de 1, 20 repeticiones).
#   4. Memoria de GPU: aumento máximo durante una inferencia (lote de
#      1) y durante un paso de entrenamiento (lote de 8, pase hacia
#      adelante y retropropagación).
#
# No guarda nada: mandame lo que imprime. Tarda un par de minutos.
# Si podés, corrélo con la GPU sin otros entrenamientos en curso.
# =====================================================================
import copy
import time

import numpy as np
import torch

RUTA_EXT_MEJOR = ("../modelos/etapa_B_Fat_Short_Unet_FACL/"
                  "B_trial0039_Fat Short Unet FACL.pth")
ARQ_EXT = dict(n_niveles=3, n_filtros_base=64,
               kernel_sizes=[5, 5, 3], dropout_ps=[0.03, 0.08, 0.26])
DIR_EXT = "../dataset"
N_CALENT, N_MED, N_CPU = 20, 200, 20
LOTE_GRANDE, LOTE_ENTR = 16, 8
N_MUESTRAS = 32

ES_CUDA = device.type == "cuda"


def _sync():
    if ES_CUDA:
        torch.cuda.synchronize()


def _ms(ts):
    ts = np.array(ts) * 1e3
    return (f"{ts.mean():8.2f} ms  (desvío {ts.std():.2f}, "
            f"mediana {np.median(ts):.2f}, mín {ts.min():.2f})")


def medir(nombre, m, ds):
    print("\n" + "=" * 68)
    n_par = sum(p.numel() for p in m.parameters())
    print(f"{nombre}: {n_par:,} parámetros "
          f"({n_par * 4 / 1e6:.2f} MB)")
    m.eval()
    rng = np.random.default_rng(0)
    ids = rng.permutation(len(ds))[:N_MUESTRAS]
    xs = [ds[int(i)][0].unsqueeze(0).to(device) for i in ids]

    # 1) GPU, lote de 1
    with torch.no_grad():
        for k in range(N_CALENT):
            m(xs[k % len(xs)])
        _sync()
        ts = []
        for k in range(N_MED):
            x = xs[k % len(xs)]
            _sync()
            t0 = time.perf_counter()
            m(x)
            _sync()
            ts.append(time.perf_counter() - t0)
    print(f"  {device.type.upper()}, lote de 1        : {_ms(ts)}")

    # 2) GPU, lote de 16 (tiempo por paso de tiempo)
    xb = torch.cat(xs[:LOTE_GRANDE])
    with torch.no_grad():
        for _ in range(5):
            m(xb)
        _sync()
        ts = []
        for _ in range(30):
            _sync()
            t0 = time.perf_counter()
            m(xb)
            _sync()
            ts.append((time.perf_counter() - t0) / LOTE_GRANDE)
    print(f"  {device.type.upper()}, lote de {LOTE_GRANDE} (c/u) : "
          f"{_ms(ts)}")
    del xb

    # 3) CPU, lote de 1
    mc = copy.deepcopy(m).to("cpu").eval()
    xc = [x.cpu() for x in xs[:4]]
    with torch.no_grad():
        mc(xc[0])
        ts = []
        for k in range(N_CPU):
            t0 = time.perf_counter()
            mc(xc[k % len(xc)])
            ts.append(time.perf_counter() - t0)
    print(f"  CPU ({torch.get_num_threads()} hilos), lote de 1 : "
          f"{_ms(ts)}")
    del mc, xc

    # 4) Memoria de GPU
    if ES_CUDA:
        torch.cuda.empty_cache()
        _sync()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        with torch.no_grad():
            m(xs[0])
        _sync()
        pico = torch.cuda.max_memory_allocated() - base
        print(f"  Memoria extra en inferencia (lote 1)      : "
              f"{pico / 1e6:8.1f} MB")

        # paso de entrenamiento: adelante + retropropagación
        banderas = [p.requires_grad for p in m.parameters()]
        for p in m.parameters():
            p.requires_grad_(True)
        m.train()
        xb = torch.cat(xs[:LOTE_ENTR])
        torch.cuda.empty_cache()
        _sync()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        out = m(xb)
        out.mean().backward()
        _sync()
        pico = torch.cuda.max_memory_allocated() - base
        print(f"  Memoria extra en entrenamiento (lote {LOTE_ENTR})  : "
              f"{pico / 1e6:8.1f} MB")
        m.zero_grad(set_to_none=True)
        m.eval()
        for p, b in zip(m.parameters(), banderas):
            p.requires_grad_(b)       # dejar el modelo como estaba
        del out, xb
        torch.cuda.empty_cache()
    else:
        print("  (sin CUDA: no se mide la memoria de GPU)")


print("PyTorch", torch.__version__, "| dispositivo:", device)
if ES_CUDA:
    print("GPU:", torch.cuda.get_device_name(0))

# Modelo final
n_par = sum(p.numel() for p in modelo.parameters())
assert n_par == 5_410_241, "`modelo` no es el ensayo 100"
ds_t = getattr(val_dataset, "dataset", val_dataset)

# Ext Clip U-Net FACL (ensayo 39)
ck = torch.load(RUTA_EXT_MEJOR, map_location=device,
                weights_only=False)
sd = ck["model_state_dict"] if (isinstance(ck, dict) and
                                "model_state_dict" in ck) else ck
ext = UNet_WRF(in_channels=next(iter(sd.values())).shape[1],
               **ARQ_EXT).to(device)
ext.load_state_dict(sd)
ds_e = TormentaDataset(split_dir=f"{DIR_EXT}/val",
                       stats_path=f"{DIR_EXT}/stats.npz")

medir("Ext Clip U-Net FACL", ext, ds_e)
medir("Temp Ext Clip U-Net FACL", modelo, ds_t)
print("\nListo: mandame toda esta salida.")
