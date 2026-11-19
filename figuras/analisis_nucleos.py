# -*- coding: utf-8 -*-
"""
Núcleos nuevos y antiguos en las estimaciones (Tabla de la sección
"Evolución temporal de las estimaciones" del capítulo de Resultados).

Uso (desde la carpeta de la monografía):
    python3 figuras/analisis_nucleos.py [UNION ENTORNO MIN_COLUMNAS]

Lee figuras/datos/resultados_validacion.npz (el mismo archivo que
figuras/generar_figuras_resultados.py) y guarda los valores de la
tabla en figuras/datos/nucleos_resultados.json.

Qué mide
--------
1) Núcleos. En cada paso de tiempo, los núcleos del campo simulado
   son los conjuntos conexos de columnas con actividad
   (LIGHTDENS > 0), uniendo en un mismo núcleo los que están
   separados por 2*UNION columnas o menos (dilatación de UNION
   columnas). Un núcleo que se superpone con uno del paso de tiempo
   anterior es su continuación; si no se superpone con ninguno, es
   un núcleo que nace. La antigüedad es el tiempo desde que nació
   (si continúa a varios, desde que nació el más antiguo). Solo se
   cuentan los núcleos con MIN_COLUMNAS columnas activas o más.
   A cada núcleo se le asigna la actividad estimada en sus columnas
   y en las que están a ENTORNO columnas o menos de su región
   dilatada (cada columna va al núcleo más cercano).
   "Nuevos": menos de 10 min de antigüedad. "Antiguos": 30 min o
   más. Se usan solo los pasos de tiempo en que hay de los dos.
   Para cada modelo: fracción de núcleos nuevos en que estima
   alguna actividad, y cociente actividad estimada / simulada
   (sumadas sobre todos los núcleos del grupo) en nuevos y antiguos.
2) Columnas. Sobre todo el conjunto de validación: fracción de las
   columnas en que la actividad simulada cesó en los 10 min previos
   (activas en alguno de los 5 pasos anteriores, inactivas ahora) en
   las que el modelo estima actividad, y fracción de las columnas en
   que la actividad simulada comenzó en este paso (activas ahora,
   inactivas en los 5 anteriores) en las que el modelo la estima.
"""
import datetime as dt
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(HERE, "datos", "resultados_validacion.npz"))
ARCHIVOS = [str(a) for a in D["archivos"]]
TIEMPOS = [dt.datetime.strptime(a[11:30], "%Y-%m-%d_%H-%M-%S")
           for a in ARCHIVOS]
REAL = D["y_real"]
MODELOS = {
    "pred_comb_clip": "Ext Clip U-Net Comb V2.1",
    "pred_ext_ultima": "Ext Clip U-Net FACL (época 140)",
    "pred_temp": "Temp Ext Clip U-Net FACL",
}
# Las estimaciones se leen una sola vez (cada acceso a D[k]
# descomprime el arreglo entero)
EST = {k: D[k] for k in MODELOS}
UNION, ENTORNO, MIN_COLUMNAS = (
    [int(x) for x in sys.argv[1:4]] if len(sys.argv) >= 4
    else (3, 8, 5))
PASO_MIN = 2          # minutos entre pasos de tiempo
NUEVO_MIN = 10        # nuevos: antigüedad < 10 min
ANTIGUO_MIN = 30      # antiguos: antigüedad >= 30 min
K = 5                 # pasos previos (10 min) para las columnas
OCHO = np.ones((3, 3), bool)


def tormentas():
    """Rangos [a, b) de índices de cada tormenta de validación."""
    cortes = [0] + [k for k in range(1, len(TIEMPOS))
                    if (TIEMPOS[k] - TIEMPOS[k - 1]).total_seconds()
                    > 120] + [len(TIEMPOS)]
    return list(zip(cortes[:-1], cortes[1:]))


def seguir_nucleos():
    """Una fila por núcleo y paso de tiempo: tormenta, paso,
    antigüedad (pasos), columnas activas, suma simulada y, por
    modelo, suma estimada."""
    filas = []
    for s, (a, b) in enumerate(tormentas()):
        ant_lab, ant_nac = None, {}
        for t in range(a, b):
            y = REAL[t]
            act = y > 0
            if not act.any():
                ant_lab, ant_nac = None, {}
                continue
            lab, n = ndi.label(
                ndi.binary_dilation(act, OCHO, iterations=UNION),
                OCHO)
            ids = np.arange(1, n + 1)
            nac = {}
            for i in ids:
                nac[i] = t
                if ant_lab is not None:
                    for p in np.unique(ant_lab[lab == i]):
                        if p:
                            nac[i] = min(nac[i], ant_nac[p])
            dist, idx = ndi.distance_transform_edt(
                lab == 0, return_indices=True)
            cerca = lab[idx[0], idx[1]]
            cerca[dist > ENTORNO] = 0
            ncol = ndi.sum(act, lab, ids)
            ssim = ndi.sum(y.astype(float), lab, ids)
            sest = [ndi.sum(EST[k][t].astype(float), cerca, ids)
                    for k in MODELOS]
            for j, i in enumerate(ids):
                if ncol[j] >= MIN_COLUMNAS:
                    filas.append([s, t - a, t - nac[i], ncol[j],
                                  ssim[j]] + [e[j] for e in sest])
            ant_lab, ant_nac = lab, nac
    return np.array(filas, float)


def columnas():
    """Por modelo: [apagadas con estimación > 0, apagadas,
    encendidas con estimación > 0, encendidas]."""
    res = {k: np.zeros(4) for k in MODELOS}
    for a, b in tormentas():
        act = REAL[a:b] > 0
        for t in range(K, b - a):
            antes = act[t - K:t].any(0)
            apag = antes & ~act[t]
            enc = act[t] & ~antes
            for k in MODELOS:
                p = EST[k][a + t] > 0
                res[k] += [p[apag].sum(), apag.sum(),
                           p[enc].sum(), enc.sum()]
    return res


def main():
    R = seguir_nucleos()
    edad = R[:, 2] * PASO_MIN
    clave = R[:, 0] * 100000 + R[:, 1]
    con_ant = np.isin(clave, clave[edad >= ANTIGUO_MIN])
    nuevo = (edad < NUEVO_MIN) & con_ant
    antiguo = (edad >= ANTIGUO_MIN) & np.isin(clave, clave[nuevo])
    intermedio = ((edad >= NUEVO_MIN) & (edad < ANTIGUO_MIN)
                  & np.isin(clave, clave[nuevo]))
    cols = columnas()
    print(f"UNION={UNION} ENTORNO={ENTORNO} "
          f"MIN_COLUMNAS={MIN_COLUMNAS}")
    print(f"Pasos de tiempo con núcleos nuevos y antiguos: "
          f"{len(set(clave[nuevo]))} de {len(TIEMPOS)}")
    print(f"Núcleos nuevos: {int(nuevo.sum())} (mediana de "
          f"{np.median(R[nuevo, 3]):.0f} columnas activas); "
          f"antiguos: {int(antiguo.sum())} (mediana de "
          f"{np.median(R[antiguo, 3]):.0f})")
    out = dict(parametros=dict(union=UNION, entorno=ENTORNO,
                               min_columnas=MIN_COLUMNAS),
               pasos=len(set(clave[nuevo])),
               n_nuevos=int(nuevo.sum()),
               n_antiguos=int(antiguo.sum()),
               columnas_nuevos=float(np.median(R[nuevo, 3])),
               columnas_antiguos=float(np.median(R[antiguo, 3])),
               modelos={})
    for j, (k, nombre) in enumerate(MODELOS.items()):
        e = R[:, 5 + j]
        rn = e[nuevo].sum() / R[nuevo, 4].sum()
        ri = e[intermedio].sum() / R[intermedio, 4].sum()
        print(f"  [{nombre}] intermedios: {int(intermedio.sum())} "
              f"(mediana {np.median(R[intermedio, 3]):.0f} columnas), "
              f"est./sim. {ri:.2f}, con actividad "
              f"{100 * (e[intermedio] > 0).mean():.1f} %, antiguos con "
              f"actividad {100 * (e[antiguo] > 0).mean():.1f} %")
        ra = e[antiguo].sum() / R[antiguo, 4].sum()
        c = cols[k]
        por_torm = []
        for s in range(len(tormentas())):
            n_, a_ = nuevo & (R[:, 0] == s), antiguo & (R[:, 0] == s)
            por_torm.append((e[n_].sum() / R[n_, 4].sum())
                            / (e[a_].sum() / R[a_, 4].sum()))
        out["modelos"][k] = dict(
            nombre=nombre,
            nuevos_detectados=float((e[nuevo] > 0).mean()),
            est_sim_nuevos=float(rn), est_sim_antiguos=float(ra),
            cociente=float(rn / ra),
            cociente_por_tormenta=[float(x) for x in por_torm],
            apagadas_con_actividad=float(c[0] / c[1]),
            encendidas_detectadas=float(c[2] / c[3]))
        print(f"\n{nombre}")
        print(f"  núcleos nuevos con alguna actividad estimada: "
              f"{100 * (e[nuevo] > 0).mean():.1f} %")
        print(f"  est./sim. en núcleos nuevos:   {rn:.2f}")
        print(f"  est./sim. en núcleos antiguos: {ra:.2f}")
        print(f"  cociente nuevos/antiguos:      {rn / ra:.2f}  "
              f"(por tormenta: "
              + ", ".join(f"{x:.2f}" for x in por_torm) + ")")
        print(f"  columnas en que la actividad cesó hace <= 10 min "
              f"y el modelo estima actividad: "
              f"{100 * c[0] / c[1]:.1f} %")
        print(f"  columnas en que la actividad comenzó y el modelo "
              f"la estima: {100 * c[2] / c[3]:.1f} %")
    if (UNION, ENTORNO, MIN_COLUMNAS) == (3, 8, 5):
        ruta = os.path.join(HERE, "datos", "nucleos_resultados.json")
        with open(ruta, "w", encoding="utf8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print("\nGuardado", ruta)


if __name__ == "__main__":
    main()
