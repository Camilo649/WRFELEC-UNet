# -*- coding: utf-8 -*-
# =====================================================================
# CELDA PARA CUALQUIER NOTEBOOK (sin GPU) - versión rápida
#
# La versión anterior le hacía un stat a cada .npz; en la carpeta
# temporal (disco de red, caché fría) eso tardó más de 2 horas.
# Esta cuenta los archivos con una sola lectura del directorio
# (os.listdir, sin stat) y mide el tamaño en disco de una MUESTRA de
# archivos por carpeta, mostrando el avance y el tiempo.
#   - memoria: exacta (nº de archivos x forma de X e y x 4 bytes);
#   - disco:   exacto si todos los archivos de la carpeta se midieron
#              (MUESTRA = None); si no, estimado = media x cantidad.
# Antes de correrla: interrumpí la celda anterior (Kernel > Interrupt).
# Salida: ../estudios/tamano_datasets.txt  <- mandame este
# =====================================================================
import os
import random
import time
from pathlib import Path

import numpy as np

CARPETAS = {'Ext (3 canales)': '../dataset',
            'Temp (9 canales)': '../dataset_temporal'}
MUESTRA = 300          # archivos medidos por carpeta (None = todos)
random.seed(0)
lineas = []


def anotar(t):
    print(t, flush=True)
    lineas.append(t)


for nombre, base in CARPETAS.items():
    anotar(f'== {nombre}: {base}')
    tot_disco = tot_mem = tot_n = 0
    exacto = True
    for split in ('train', 'val'):
        t0 = time.time()
        carpeta = Path(base, split)
        nombres = sorted(n for n in os.listdir(carpeta)
                         if n.endswith('.npz'))
        print(f'  {split}: {len(nombres)} archivos listados en '
              f'{time.time() - t0:.1f} s', flush=True)
        elegidos = (nombres if MUESTRA is None or
                    len(nombres) <= MUESTRA
                    else random.sample(nombres, MUESTRA))
        t0 = time.time()
        tam = []
        for k, n in enumerate(elegidos):
            tam.append(os.stat(carpeta / n).st_size)
            if k % 50 == 0:
                print(f'    stat {k}/{len(elegidos)} '
                      f'({time.time() - t0:.1f} s)', flush=True)
        medido_todo = len(elegidos) == len(nombres)
        exacto &= medido_todo
        disco = (sum(tam) if medido_todo
                 else np.mean(tam) * len(nombres))
        with np.load(carpeta / nombres[0]) as d:
            fx, fy = tuple(d['X'].shape), tuple(d['y'].shape)
            mem = len(nombres) * (d['X'].size + d['y'].size) * 4
        anotar(f"  {split:5s}: {len(nombres):6d} muestras | X {fx}, "
               f"y {fy} | disco {disco / 1e6:9.1f} MB"
               f"{'' if medido_todo else ' (estimado)'} | memoria "
               f"float32 {mem / 1e6:9.1f} MB")
        tot_disco += disco
        tot_mem += mem
        tot_n += len(nombres)
    anotar(f'  total: {tot_n} muestras | disco {tot_disco / 1e6:.1f} '
           f"MB ({tot_disco / 2**30:.2f} GiB)"
           f"{'' if exacto else ' (estimado)'} | memoria float32 "
           f'{tot_mem / 1e6:.1f} MB ({tot_mem / 2**30:.2f} GiB)')
Path('../estudios').mkdir(exist_ok=True)
Path('../estudios/tamano_datasets.txt').write_text('\n'.join(lineas),
                                                   encoding='utf-8')
print('Guardado ../estudios/tamano_datasets.txt')
