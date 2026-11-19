# WRFELEC-UNet

U-Nets que estiman la densidad de descargas eléctricas simulada por
WRF-ELEC (variable `LIGHTDENS`) a partir de tres variables de salida del
modelo WRF: reflectividad máxima de la columna (`MDBZ`), contenido de hielo
en la región de fase mixta y temperatura del tope de la nube. Los modelos se
entrenan con la función de pérdida FACL
([Yan et al., 2024](https://arxiv.org/abs/2410.23159)).

*U-Nets that estimate the lightning density simulated by WRF-ELEC from three
WRF output variables (maximum column reflectivity, ice content in the
mixed-phase region and cloud-top temperature), trained with the FACL loss.*

Este repositorio acompaña el Trabajo Especial de la Licenciatura en Ciencias
de la Computación (FAMAF, Universidad Nacional de Córdoba):

> Gutiérrez, C. (2026). *Estimación de la Actividad Eléctrica Simulada por
> WRF-ELEC a Partir de Variables del Modelo WRF Mediante Redes Neuronales
> Convolucionales*. Director: Dra. Melina Yasmín Luque.

## Modelos incluidos

Los nombres siguen la nomenclatura de la tesis (Capítulo 5, sección
*Nomenclatura*): *Temp* = entrada temporal, *Ext* = conjunto de datos
extendido, *Clip* = `LIGHTDENS` restringido a 150.

| Modelo | Carpeta | Parámetros | Épocas (mejor) |
|---|---|---|---|
| Ext U-Net Comb V2.1 | `results/ext_u-net_comb_v2.1` | 8.113.665 | 37 (17) |
| Ext Clip U-Net FACL | `results/ext_clip_u-net_facl` | 9.555.457 | 140 (8) |
| Temp Ext Clip U-Net FACL | `results/temp_ext_clip_u-net_facl` | 5.410.241 | 60 (47) |

Los dos últimos son los que se analizan en el capítulo de resultados.
Para la Ext Clip U-Net FACL, `ultima_epoca` es la época evaluada en la tesis
(la 140) y `mejor_epoca_Lval` corresponde a la época de menor pérdida de
validación (la 8).

`results/adicionales/` contiene un modelo que no se discute en la tesis:
la Temp Ext Clip U-Net FACL entrenada con la FACL de Juan (pérdida de
validación ½FAL + ½FCL), usada para la comparación de las búsquedas de
hiperparámetros.

## Contenido

```
modulos/     tormentaDataset.py (Dataset de PyTorch usado por los notebooks)
notebooks/   Datasets, entrenamientos y optimizaciones (sin outputs)
figuras/     Scripts que generan las figuras y tablas de la tesis
results/
  <modelo>/
    historial_entrenamiento.txt   pérdidas por época
    curva_error.png               curvas de pérdida
    metricas/                     MSE, RMSE, MAE, SSIM, POD, FAR, CSI, FSS, residuos
    distribucion_curvas/          distribución real vs. estimada
    animaciones/                  reconstrucciones de 4 tormentas (MP4)
```

Las cuatro tormentas animadas son las de los días 14/12/2018, 29/01/2019,
09/02/2019 y 04/03/2019.

## Notebooks

| Notebook | Contenido |
|---|---|
| `Datasets` | Análisis, filtrado y construcción de los datasets (Fases 1 a 5, incluido el temporal) |
| `Pruebas` | Bob, Baseline, U-Net y variantes con MSE/WMSE (conjunto inicial) |
| `Entrenamiento_U-Net_Comb_V1` | U-Net Comb V1 y V1.1 (pérdida híbrida, conjunto inicial) |
| `Entrenamiento_U-Net_Comb_V2_AvgPool` | U-Net Comb V2 con la prueba de *average pooling* |
| `Entrenamiento_U-Net_Fer-Net` | Esquemas de ponderación asimétrica de la FACL (Fer-Net, Net-Fer) |
| `Entrenamiento_Ext_U-Net_Comb_V2.1` | Ext U-Net Comb V2.1: entrenamiento, importancia de variables y optimización |
| `Entrenamiento_Ext_Clip_U-Net_FACL` | Ext Clip U-Net FACL: entrenamiento, importancia de variables y optimización |
| `Entrenamiento_Temp_Ext_Clip_U-Net_FACL` | Temp Ext Clip U-Net FACL (último modelo) |
| `Entrenamiento_Ext_Clip_U-Net_Comb_V2.1` | Ext Clip U-Net Comb V2.1: evaluación del mejor ensayo de su optimización |
| `recopilar_resultados_comb.py` | Reúne métricas, curvas, historial y animaciones de la Comb V2.1 en la estructura de `results/` |
| `Graficas` | Figuras de la tesis |
| `auxiliares/` | Preliminares (`modelos_preliminares`, en Keras), exploración de un `wrfout`, validación de las simulaciones nuevas y lectura de los estudios de Optuna |

Los notebooks se ejecutaron en el JupyterHub del CCAD. Los dos últimos
modelos (Ext Clip U-Net FACL y Temp Ext Clip U-Net FACL) se entrenaron y
optimizaron en el clúster Clementina con los scripts de `clementina/`
(trabajos de Slurm); los notebooks `Ext_Clip` y `Temp_Ext_Clip` documentan su diseño y
análisis.

## Scripts de Clementina

`clementina/` contiene el código con el que se hicieron las búsquedas de
hiperparámetros (Optuna, etapas A y B) y los entrenamientos finales en el clúster
Clementina (GPU Intel Data Center Max, entorno `torch_xpu`, trabajos de Slurm):
`datos.py`, `modelo.py`, las funciones de pérdida (`facl_oficial.py`,
`perdida_comb.py`), `objetivo_*.py`, `worker_*.py`, `train_*.py` y los
`sbatch_*.sh`. La optimización de la Ext Clip U-Net Comb V2.1 usa
`objetivo_comb.py`, `worker_comb.py` y `lanzar_comb.sh`, y
`exportar_checkpoint_comb.py` convierte su mejor ensayo al formato del notebook. Los sufijos `_oficial` y `_temporal` indican la FACL de Yan et al.
con el MAE normalizado como criterio de validación y el conjunto temporal,
respectivamente.

## Pesos

`pesos/ext_clip_u-net_facl_trial39.pth`: checkpoint de la Ext Clip U-Net FACL
(trial 39, 9.555.457 parámetros, 140 épocas). Contiene `model_state_dict`
(última época, la evaluada en la tesis), `best_model_state_dict` (mejor época
según la pérdida de validación) y el historial de entrenamiento. Se carga con
`torch.load(..., weights_only=True)` y la arquitectura de `clementina/modelo.py`.

`pesos/temp_ext_clip_u-net_facl_best_epoca47.pt`, `..._last_epoca60.pt` y
`..._curva.csv`: Temp Ext Clip U-Net FACL (ensayo 100 de la búsqueda
`etapa_B_oficial`, 5.410.241 parámetros). `best` es la época 47 (la evaluada
en la tesis, MAE normalizado de validación 0,00587) y `last`, la 60. Cada `.pt`
guarda `modelo` (pesos) y las métricas de validación de esa época; se carga con
`torch.load(..., weights_only=True)`. La arquitectura se reconstruye con
`clementina/modelo.py` y los hiperparámetros del ensayo 100.

## Hiperparámetros de los dos últimos modelos

| | Ext Clip U-Net FACL | Temp Ext Clip U-Net FACL |
|---|---|---|
| Ensayo de la etapa A (arquitectura) | 106 | 118 |
| Niveles / filtros | 3 / 64, 128, 256 | 3 / 32, 64, 128 |
| Tamaños de kernel | 5, 5, 3 | 5, 5, 5 |
| Ensayo de la etapa B (entrenamiento) | 39 | 100 |
| Tasa de aprendizaje inicial | 1,22·10⁻⁴ | 5,99·10⁻⁴ (0,000599407) |
| Tamaño de lote | 8 | 8 |
| α (FACL) | 0,189 | 0,229 (0,229207) |
| Épocas | 140 | 60 |
| Criterio de validación | L_val = ½FAL + ½FCL | MAE normalizado |
| Época evaluada | 140 | 47 |

El ensayo 100 de la búsqueda `etapa_B_oficial` terminó con pérdida de
validación 0,005871 (época 47), 5.410.241 parámetros y 67.500 pasos de
entrenamiento.

Nota sobre la Ext Clip U-Net FACL: `pesos/ext_clip_u-net_facl_trial39.pth` es el
entrenamiento original del ensayo 39 en Clementina (mejor época según L_val:
la 5). Las curvas de FAL y FCL y las estimaciones de la tesis provienen de un
reentrenamiento con las mismas semillas e hiperparámetros (mejor época: la 8),
cuyos pesos no se conservaron.

## Datos

El conjunto de datos (salidas de las simulaciones WRF y WRF-ELEC, 23 días de
tormenta sobre la provincia de Córdoba, resolución horizontal de 2 km y una
salida cada 2 minutos) no está incluido por su tamaño. Su construcción se
describe en `notebooks/Datasets.ipynb` y en el Capítulo 4 de la tesis.

## Reproducibilidad

- Notebooks: Python 3.12, PyTorch 2.6.0 con CUDA 12.4 (`requirements.txt`). Scripts de Clementina: entorno
  `torch_xpu` (PyTorch para GPU Intel).
- Los notebooks conservan rutas de la máquina donde se entrenaron
  (`/home/gcamilo/...`, `drive/MyDrive/...`): hay que ajustarlas.
- Los notebooks esperan la estructura `../dataset`, `../modelos`, `../estudios`, etc.
  relativa a `notebooks/`, y usan `modulos/tormentaDataset.py`.
- Los notebooks de `auxiliares/modelos_preliminares` usan TensorFlow/Keras; el
  resto, PyTorch.

## Licencia

Código: [MIT](LICENSE). Resultados (figuras, métricas, animaciones y
tablas): [CC BY-NC 4.0](LICENSE-results.md), igual que la tesis.

## Cita

Ver `CITATION.cff`.
