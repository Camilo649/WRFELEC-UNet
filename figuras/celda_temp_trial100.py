# -*- coding: utf-8 -*-
# Celda para el final de Entrenamiento_Temp_Ext_Clip_U-Net_FACL.ipynb
# (necesita `modelo` = ensayo 100, `val_dataset`, `device`).
# Genera temp_trial100_instante.npz con la predicción del modelo
# final para el instante de la figura de comparación.
import numpy as np
import torch

ARCHIVO = "wrfout_d01_2018-12-14_01-00-00.npz"   # 14/12/2018 01:00 UTC

# Chequeo: tiene que ser el modelo final
n_par = sum(p.numel() for p in modelo.parameters())
print("Parámetros:", f"{n_par:,}", "(esperado: 5,410,241)")
assert n_par == 5_410_241, "`modelo` no es el ensayo 100"

ds = getattr(val_dataset, "dataset", val_dataset)
nombres = [f.name for f in ds.files]
i = nombres.index(ARCHIVO)

modelo.eval()
X, y = val_dataset[i]
with torch.no_grad():
    p = modelo(X.unsqueeze(0).to(device)).cpu().squeeze()
pred = np.round(ds.desnormalizar_y(p).numpy()).astype(np.float32)
real = np.round(ds.desnormalizar_y(y.squeeze()).numpy()
                ).astype(np.float32)

m = (real > 0) | (pred > 0)
csi = float(((real > 0) & (pred > 0)).sum() / m.sum()) \
    if m.sum() else float("nan")
print(f"{ARCHIVO}: real max {real.max():.0f}, "
      f"pred max {pred.max():.0f}, CSI {csi:.4f}")

np.savez("temp_trial100_instante.npz",
         nombre="Temp Ext Clip U-Net FACL",
         archivo=ARCHIVO, y_real=real, pred=pred, csi=csi,
         X=X.numpy())
print("Guardado temp_trial100_instante.npz")
