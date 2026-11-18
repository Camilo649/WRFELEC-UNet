# -*- coding: utf-8 -*-
# =====================================================================
# CELDA PARA LA NOTEBOOK (cualquiera que tenga torch importado).
#
# Qué hace: abre todos los checkpoints (.pth) de ../modelos y busca
# lo que quedó guardado adentro: hiperparámetros, coeficientes de la
# pérdida, épocas, tiempos y el historial de entrenamiento. Sirve para
# completar los "?" de la Tabla 4.6 y para revisar los tiempos por
# época raros (V1.1, V1 y V2) de la Tabla 4.5.
#
# Salida: se imprime y se guarda en inspeccion_checkpoints.txt.
# Mandame ese .txt.
# =====================================================================
import glob
import os
import time
import numpy as np
import torch

CARPETA = "../modelos"
CLAVES = ("lr", "learn", "batch", "alpha", "beta", "peso", "prob",
          "slope", "norm", "epoca", "epoch", "tiempo", "time",
          "durac", "segundos", "loss", "func", "arq", "config",
          "hiper", "param", "patience", "warm", "optim", "sched")

salida = []


def p(*a):
    t = " ".join(str(x) for x in a)
    print(t)
    salida.append(t)


def resumen(v, prof=0):
    """Texto corto de un valor: escalares enteros, listas con
    longitud, tensores con forma, dicts con sus claves."""
    if isinstance(v, (int, float, str, bool, type(None))):
        return repr(v)[:120]
    if isinstance(v, (np.integer, np.floating)):
        return repr(v.item())
    if torch.is_tensor(v):
        return (f"tensor{tuple(v.shape)}" if v.numel() > 1
                else repr(v.item()))
    if isinstance(v, (list, tuple)):
        if len(v) and all(isinstance(x, (int, float)) for x in v):
            a = np.array(v, dtype=float)
            return (f"lista de {len(v)} números "
                    f"[{a[0]:.4g} ... {a[-1]:.4g}] "
                    f"suma={a.sum():.6g}")
        return f"lista de {len(v)} elementos"
    if isinstance(v, dict):
        if prof >= 1:
            return f"dict de {len(v)} claves"
        return "{" + ", ".join(f"{k}: {resumen(x, prof + 1)}"
                               for k, x in list(v.items())[:25]) + "}"
    return type(v).__name__


archivos = sorted(glob.glob(f"{CARPETA}/**/*.pth", recursive=True))
p(f"{len(archivos)} checkpoints en {CARPETA}\n")

for ruta in archivos:
    p("=" * 70)
    p(ruta)
    st = os.stat(ruta)
    p("  modificado:", time.strftime("%Y-%m-%d %H:%M",
                                     time.localtime(st.st_mtime)),
      f"| {st.st_size / 1e6:.1f} MB")
    try:
        ck = torch.load(ruta, map_location="cpu", weights_only=False)
    except Exception as e:
        p("  !! no se pudo abrir:", e)
        continue
    if not isinstance(ck, dict):
        p("  (state_dict directo, sin metadatos)")
        continue
    for k, v in ck.items():
        if k in ("model_state_dict", "optimizer_state_dict",
                 "scheduler_state_dict"):
            p(f"  {k}: (pesos/estado, {len(v)} entradas)")
            continue
        marca = "*" if any(c in k.lower() for c in CLAVES) else " "
        p(f" {marca}{k}: {resumen(v)}")
        # historiales: dict de listas o lista de dicts
        if isinstance(v, dict) and v and all(
                isinstance(x, (list, tuple)) for x in v.values()):
            for kk, vv in v.items():
                if len(vv):
                    p(f"        .{kk}: {resumen(vv)}")
p("\nFIN")

with open("inspeccion_checkpoints.txt", "w", encoding="utf8") as f:
    f.write("\n".join(salida))
print("\nGuardado en inspeccion_checkpoints.txt")
