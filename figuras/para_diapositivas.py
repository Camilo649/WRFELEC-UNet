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
CONSTRAINED = True      # diseño automático en vez de fracciones fijas
RALEAR = True           # quitar marcas que se pisan

# figura: (script, escala)
ESCALAS = {
    "ejemplo_problema": ("generar_figuras_instante.py", 0.72),
    "ejemplo_variables": ("generar_figuras_instante.py", 0.62),
    "deltas_hielo": ("generar_figuras_instante.py", 0.42),
    "limitacion_cnn_base": ("generar_figuras_instante.py", 0.61),
    "imp_corrida1": ("generar_figuras_datos.py", 0.78),
    "comparacion_reconstrucciones_tormentas":
        ("generar_figuras_modelos.py", 0.40),
    "series_tormentas": ("generar_figuras_resultados.py", 0.85),
    "distribucion_lightdens": ("generar_figuras_resultados.py", 0.58),
    "bordes_dominio": ("generar_figura_bordes.py", 0.80),
    "lma_vs_wrf": ("generar_figura_lma.py", 0.64),
    "campo_efectivo": ("generar_figura_campo_efectivo_tres.py", 0.78),
    "Cumulonimbus": ("generar_cumulonimbus.py", 0.65),
    "redes_unet": ("generar_unet_final.py", 0.72),
}


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
    """Quita marcas de los ejes cuyos rótulos se pisan."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for ax in fig.axes:
        for eje in (ax.xaxis, ax.yaxis):
            marcas = [(t.get_loc(), t.label1.get_text()) for t in
                      eje.get_major_ticks() if t.label1.get_visible()
                      and t.label1.get_text()]
            if len(marcas) < 3:
                continue
            for paso in (2, 3, 4):
                cajas = [t.label1.get_window_extent(rend).expanded(
                    1.08, 1.0) for t in eje.get_major_ticks()
                    if t.label1.get_visible() and t.label1.get_text()]
                if not any(a.overlaps(b) for i, a in enumerate(cajas)
                           for b in cajas[i + 1:]):
                    break
                sub = marcas[::paso]
                if marcas[-1] not in sub and paso == 2:
                    sub = sub + [marcas[-1]]
                eje.set_ticks([m[0] for m in sub])
                eje.set_ticklabels([m[1] for m in sub])
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
        iniciar(self, *a, figsize=(figsize[0] * escala,
                                   figsize[1] * escala), **k)

    def savefig(self, fname, *a, **k):
        base = os.path.basename(str(fname))
        if os.path.splitext(base)[0] not in solo:
            return None
        _oscurecer(self)
        if RALEAR:
            _ralear(self)
        return guardar(self, os.path.join(salida, base), *a, **k)

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
        script, escala = ESCALAS[n]
        grupos.setdefault((script, escala), set()).add(n)
    for (script, escala), solo in grupos.items():
        print(f"== {script} (escala {escala}): {', '.join(sorted(solo))}")
        correr(script, escala, solo, salida)


main()
