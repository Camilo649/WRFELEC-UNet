# -*- coding: utf-8 -*-
# =====================================================================
# CELDA PARA CUALQUIER NOTEBOOK (no necesita GPU ni más celdas)
#
# Qué hace: mide el tamaño de los dos conjuntos de datos del
# capítulo 6 (el extendido, de 3 canales, y el temporal, de 9
# canales), en entrenamiento y validación, de dos maneras:
#   - en disco: suma de los .npz de cada carpeta;
#   - en memoria: lo que ocuparía todo el conjunto cargado como
#     tensores float32 (entrada + variable objetivo), que es como los
#     lee TormentaDataset.
# Correla desde la carpeta de los notebooks (las rutas son las de las
# notebooks: ../dataset y ../dataset_temporal).
#
# Salida: ../estudios/tamano_datasets.txt  <- mandame este
# =====================================================================
from pathlib import Path

import numpy as np

CARPETAS = {'Ext (3 canales)': '../dataset',
            'Temp (9 canales)': '../dataset_temporal'}
lineas = []


def anotar(t):
    print(t)
    lineas.append(t)


for nombre, base in CARPETAS.items():
    anotar(f'== {nombre}: {base}')
    tot_disco = tot_mem = tot_n = 0
    for split in ('train', 'val'):
        archivos = sorted(Path(base, split).glob('*.npz'))
        disco = sum(f.stat().st_size for f in archivos)
        d = np.load(archivos[0])
        mem = len(archivos) * (d['X'].size + d['y'].size) * 4
        anotar(f"  {split:5s}: {len(archivos):6d} muestras | X "
               f"{tuple(d['X'].shape)} {d['X'].dtype}, y "
               f"{tuple(d['y'].shape)} {d['y'].dtype} | disco "
               f"{disco / 1e6:9.1f} MB | memoria float32 "
               f"{mem / 1e6:9.1f} MB")
        tot_disco += disco
        tot_mem += mem
        tot_n += len(archivos)
    anotar(f'  total: {tot_n} muestras | disco {tot_disco / 1e6:.1f} MB'
           f' ({tot_disco / 2**30:.2f} GiB) | memoria float32 '
           f'{tot_mem / 1e6:.1f} MB ({tot_mem / 2**30:.2f} GiB)')
Path('../estudios').mkdir(exist_ok=True)
Path('../estudios/tamano_datasets.txt').write_text('\n'.join(lineas),
                                                   encoding='utf-8')
