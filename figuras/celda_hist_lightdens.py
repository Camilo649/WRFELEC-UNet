# Celda para Datasets.ipynb: histograma de LIGHTDENS en el conjunto de
# entrenamiento (valores ya restringidos en 150), para la figura de la
# Sección "Rangos y Normalización". Guarda figuras/datos/hist_lightdens.npz.
import glob, os
import numpy as np
from tqdm.auto import tqdm

archivos = sorted(glob.glob(f"{OUTPUT_DIR}/train/*.npz"))   # dataset final
EDGES_Y = np.arange(0, 151, 1.0)                  # (a) y en (0, 150]
EDGES_L = np.linspace(-2.31, 5.02, 75)            # (b) ln(y + 0,1)

counts_y = np.zeros(len(EDGES_Y) - 1, dtype=np.int64)
counts_l = np.zeros(len(EDGES_L) - 1, dtype=np.int64)
n_cero = n_total = 0
for f in tqdm(archivos):
    y = np.load(f)["y"].astype(np.float64)
    act = y[y > 0]
    n_total += y.size
    n_cero += y.size - act.size
    counts_y += np.histogram(act, bins=EDGES_Y)[0]
    counts_l += np.histogram(np.log(act + 0.1), bins=EDGES_L)[0]

os.makedirs("figuras/datos", exist_ok=True)
np.savez("figuras/datos/hist_lightdens.npz",
         edges_y=EDGES_Y, counts_y=counts_y,
         edges_l=EDGES_L, counts_l=counts_l,
         n_cero=n_cero, n_total=n_total)
print("columnas con actividad:", n_total - n_cero, "de", n_total)
print("máx y:", EDGES_Y[-1], "| activas por intervalo (a):", counts_y[:10])
print("fracción de activas con y < 5:",
      counts_y[:5].sum() / counts_y.sum(),
      "| y < 10:", counts_y[:10].sum() / counts_y.sum())
