"""Regenera las figuras con la letra más grande para las diapositivas.

Las figuras de la tesis se dibujan para un ancho de página de unos
14 cm. En una diapositiva entran en una columna de 5 a 8 cm, así que
la letra de 6 a 8 pt queda en 2 a 4 pt. Este lanzador corre los
mismos scripts de figuras sin modificarlos, pero achica el lienzo de
cada figura por una escala (la letra no cambia, así que queda más
grande en proporción) y guarda el resultado en otra carpeta, sin
tocar img/.

Uso:
    python3 figuras/para_diapositivas.py            # todas
    python3 figuras/para_diapositivas.py nombre...  # solo esas
    python3 figuras/para_diapositivas.py --salida DIR

La tabla ESCALAS dice, para cada figura, su script y la escala. Una
escala de 0,5 deja la letra al doble de tamaño relativo. Se ajusta
mirando la figura al tamaño con que entra en la diapositiva.
"""
import os
import runpy
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

HERE = os.path.dirname(os.path.abspath(__file__))
AJUSTE = {}             # sy: factor extra de alto; hspace
CONSTRAINED = True      # diseño automático en vez de fracciones fijas
RALEAR = True           # quitar marcas que se pisan

# figura: (script, escala[, diseño automático = True])
ESCALAS = {
    "ejemplo_problema": ("generar_figuras_instante.py", 0.72),
    "ejemplo_variables": ("generar_figuras_instante.py", 0.7, False),
    "deltas_hielo": ("generar_figuras_instante.py", 0.42),
    "limitacion_cnn_base": ("generar_figuras_instante.py", 0.61),
    "imp_corrida1": ("generar_figuras_datos.py", 0.78),
    "comparacion_reconstrucciones_tormentas":
        ("generar_figuras_modelos.py", 0.40),
    "series_tormentas": ("generar_figuras_resultados.py", 0.85, False),
    "distribucion_lightdens": ("generar_figuras_resultados.py", 0.58),
    "bordes_dominio": ("generar_figura_bordes.py", 0.80),
    "lma_vs_wrf": ("generar_figura_lma.py", 0.64),
    "campo_efectivo": ("generar_figura_campo_efectivo_tres.py", 0.78),
    "Cumulonimbus": ("generar_cumulonimbus.py", 0.65),
    "facl_espectros": ("generar_figuras_facl.py", 0.66),
    "facl_cronograma": ("generar_figuras_facl.py", 0.82, False),
    "redes_unet": ("generar_unet_final.py", 0.72),
}


# ajustes propios de una figura (alto relativo, hspace)
AJUSTES = {"ejemplo_variables": {"sy": 0.92, "hspace": 0.62},
           "series_tormentas": {"hspace": 0.42}}


def _oscurecer(fig):
    """Los grises claros de los rótulos no se ven en un proyector."""
    from matplotlib.colors import to_rgb
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        try:
            r, g, b = to_rgb(t.get_color())
        except ValueError:
            continue
        luz = 0.3 * r + 0.59 * g + 0.11 * b
        if luz > 0.35 and max(r, g, b) - min(r, g, b) < 0.12:
            t.set_color("#3d3d3a")


def _ralear(fig):
    """Quita marcas de los ejes cuyos rótulos se pisan. Solo cuenta
    las marcas dentro del rango visible y no cambia los límites."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for ax in fig.axes:
        for eje, lim in ((ax.xaxis, ax.get_xlim),
                         (ax.yaxis, ax.get_ylim)):
            a0, a1 = sorted(lim())
            tol = 1e-9 * (a1 - a0)

            def visibles():
                return [t for t in eje.get_major_ticks()
                        if a0 - tol <= t.get_loc() <= a1 + tol
                        and t.label1.get_visible()
                        and t.label1.get_text()]

            marcas = [(t.get_loc(), t.label1.get_text())
                      for t in visibles()]
            if len(marcas) < 3:
                continue
            for paso in (2, 3, 4):
                cajas = [t.label1.get_window_extent(rend).expanded(
                    1.45, 1.0) for t in visibles()]
                if not any(a.overlaps(b) for i, a in enumerate(cajas)
                           for b in cajas[i + 1:]):
                    break
                sub = marcas[::paso]
                if marcas[-1] not in sub and paso == 2:
                    sub = sub + [marcas[-1]]
                ant = (ax.get_xlim(), ax.get_ylim())
                eje.set_ticks([m[0] for m in sub])
                eje.set_ticklabels([m[1] for m in sub])
                ax.set_xlim(ant[0])
                ax.set_ylim(ant[1])
                fig.canvas.draw()


def correr(script, escala, solo, salida):
    """Corre un script de figuras guardando solo las de 'solo'."""
    guardar = Figure.savefig
    iniciar = Figure.__init__

    def init(self, *a, figsize=None, **k):
        if figsize is None:
            figsize = matplotlib.rcParams["figure.figsize"]
        if CONSTRAINED and "layout" not in k \
                and "constrained_layout" not in k:
            k["layout"] = "constrained"
        iniciar(self, *a, figsize=(
            figsize[0] * escala,
            figsize[1] * escala * AJUSTE.get("sy", 1)), **k)

    def savefig(self, fname, *a, **k):
        base = os.path.basename(str(fname))
        if os.path.splitext(base)[0] not in solo:
            return None
        _oscurecer(self)
        eng = self.get_layout_engine()
        if eng is not None and "h_pad" in AJUSTE:
            eng.set(h_pad=AJUSTE["h_pad"], hspace=0.0)
        if RALEAR:
            _ralear(self)
        return guardar(self, os.path.join(salida, base), *a, **k)

    from matplotlib.gridspec import GridSpec
    gs_init = GridSpec.__init__

    def gsinit(self, *a, **k):
        if "hspace" in AJUSTE:
            k["hspace"] = AJUSTE["hspace"]
        gs_init(self, *a, **k)

    GridSpec.__init__ = gsinit
    Figure.__init__ = init
    Figure.savefig = savefig
    viejo = sys.argv
    try:
        sys.argv = [script]
        runpy.run_path(os.path.join(HERE, script), run_name="__main__")
    finally:
        sys.argv = viejo
        Figure.__init__ = iniciar
        Figure.savefig = guardar
        GridSpec.__init__ = gs_init


def main():
    args = sys.argv[1:]
    salida = os.path.join(HERE, "..", "img_diapositivas")
    if "--salida" in args:
        i = args.index("--salida")
        salida = args[i + 1]
        del args[i:i + 2]
    nombres = args or list(ESCALAS)
    os.makedirs(salida, exist_ok=True)
    grupos = {}
    for n in nombres:
        script, escala, *auto = ESCALAS[n]
        extra = tuple(sorted(AJUSTES.get(n, {}).items()))
        grupos.setdefault((script, escala, extra, *auto),
                          set()).add(n)
    global CONSTRAINED, AJUSTE
    for (script, escala, extra, *auto), solo in grupos.items():
        AJUSTE = dict(extra)
        CONSTRAINED = auto[0] if auto else True
        print(f"== {script} (escala {escala}): {', '.join(sorted(solo))}")
        correr(script, escala, solo, salida)


main()
