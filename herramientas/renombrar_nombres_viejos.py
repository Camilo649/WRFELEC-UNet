#!/usr/bin/env python3
"""Pasa los archivos generados con los nombres internos del desarrollo
(Fat Short Unet FACL, Memo ..., etc.) a los nombres de la tesis.

Hace tres cosas bajo las carpetas que se le indican:
  1. Renombra archivos y carpetas cuyo nombre contiene un nombre viejo
     (checkpoints .pth/.pt, estudios .db, historiales, etapa_B_*).
  2. Actualiza el nombre del estudio guardado dentro de los .db de Optuna
     (sqlite, tabla `studies`).
  3. Reemplaza el nombre del estudio dentro de los journals de Optuna
     (`optuna_*.log`, texto JSON por línea).

Por defecto solo muestra qué haría. Con --aplicar lo hace, y antes guarda
una copia de cada .db y journal que modifica (extensión .bak).

Uso:
    python3 herramientas/renombrar_nombres_viejos.py ../modelos ../estudios
    python3 herramientas/renombrar_nombres_viejos.py ../modelos ../estudios --aplicar
"""
import argparse
import os
import shutil
import sqlite3

# El orden importa: primero los nombres más largos o más específicos.
EQUIVALENCIAS = [
    ("etapa_B_Memo_Fat_Short_Unet_FACL", "etapa_B_Temp_Ext_Clip_U-Net_FACL"),
    ("etapa_B_Fat_Short_Unet_FACL", "etapa_B_Ext_Clip_U-Net_FACL"),
    ("Memo Fat Short Unet FACL", "Temp Ext Clip U-Net FACL"),
    ("Fat Short Unet Comb V2.2", "Ext U-Net Comb V2.1"),
    ("Fat Short Unet FACL", "Ext Clip U-Net FACL"),
    ("Fat unet Comb V2.2", "Ext U-Net Comb V2.1"),
    ("unet Comb V2 avgPool", "U-Net Comb V2 AvgPool"),
    ("unet Comb V2.2", "U-Net Comb V2"),
    ("unet Comb V1.1", "U-Net Comb V1.1"),
]


def nuevo(texto):
    for viejo, nv in EQUIVALENCIAS:
        texto = texto.replace(viejo, nv)
    return texto


def respaldar(ruta, aplicar):
    if aplicar and not os.path.exists(ruta + ".bak"):
        shutil.copy2(ruta, ruta + ".bak")


def actualizar_db(ruta, aplicar):
    con = sqlite3.connect(ruta)
    try:
        filas = con.execute("SELECT study_id, study_name FROM studies").fetchall()
    except sqlite3.DatabaseError:
        con.close()
        return
    cambios = [(i, n, nuevo(n)) for i, n in filas if nuevo(n) != n]
    for i, n, nv in cambios:
        print(f"  estudio en {ruta}: {n!r} -> {nv!r}")
    con.close()
    if cambios and aplicar:
        respaldar(ruta, True)
        con = sqlite3.connect(ruta)
        con.executemany("UPDATE studies SET study_name = ? WHERE study_id = ?",
                        [(nv, i) for i, _, nv in cambios])
        con.commit()
        con.close()


def actualizar_journal(ruta, aplicar):
    with open(ruta, "r", encoding="utf-8") as f:
        texto = f.read()
    if nuevo(texto) == texto:
        return
    print(f"  nombre de estudio en el journal {ruta}")
    if aplicar:
        respaldar(ruta, True)
        with open(ruta, "w", encoding="utf-8") as f:
            f.write(nuevo(texto))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("carpetas", nargs="+")
    ap.add_argument("--aplicar", action="store_true",
                    help="hace los cambios (sin esta opción solo los muestra)")
    a = ap.parse_args()

    for raiz in a.carpetas:
        # 1) contenido de .db y journals, antes de renombrar los archivos
        for d, _, archivos in os.walk(raiz):
            for nombre in archivos:
                ruta = os.path.join(d, nombre)
                if nombre.endswith(".db"):
                    actualizar_db(ruta, a.aplicar)
                elif nombre.startswith("optuna_") and nombre.endswith(".log"):
                    actualizar_journal(ruta, a.aplicar)
        # 2) nombres de archivos y carpetas, de adentro hacia afuera
        for d, carpetas, archivos in os.walk(raiz, topdown=False):
            for nombre in archivos + carpetas:
                nv = nuevo(nombre)
                if nv == nombre or nombre.endswith(".bak"):
                    continue
                origen, destino = os.path.join(d, nombre), os.path.join(d, nv)
                if os.path.exists(destino):
                    print(f"  OMITIDO (ya existe el destino): {destino}")
                    continue
                print(f"  {origen} -> {nv}")
                if a.aplicar:
                    os.rename(origen, destino)
    if not a.aplicar:
        print("\nNo se cambió nada. Repetí el comando con --aplicar para hacerlo.")


if __name__ == "__main__":
    main()
