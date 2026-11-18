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

## Datos

El conjunto de datos (salidas de las simulaciones WRF y WRF-ELEC, 23 días de
tormenta sobre la provincia de Córdoba, resolución horizontal de 2 km y una
salida cada 2 minutos) no está incluido por su tamaño. Su construcción se
describe en `notebooks/Datasets.ipynb` y en el Capítulo 4 de la tesis.

## Reproducibilidad

- Las dependencias están en `requirements.txt` (sin versiones fijas todavía).
- Los notebooks conservan rutas de la máquina donde se entrenaron
  (`/home/gcamilo/...`, `drive/MyDrive/...`): hay que ajustarlas.
- Los notebooks importan el módulo local `modulos.tormentaDataset`.

## Licencia

Código: [MIT](LICENSE). Resultados (figuras, métricas, animaciones y
tablas): [CC BY-NC 4.0](LICENSE-results.md), igual que la tesis.

## Cita

Ver `CITATION.cff`.
