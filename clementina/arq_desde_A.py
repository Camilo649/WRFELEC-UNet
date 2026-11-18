"""Lee el journal de la Etapa A y deja la arquitectura de la Etapa B en estudios/arq_comb_A.json.
Regla por defecto (la de la Comb V2): la arquitectura del mejor ensayo completado, con los dropouts redondeados a
2 decimales. Con --promedio5 se usa la de la Ext Clip U-Net FACL: topología del mejor ensayo y dropout promedio (a 2
decimales) de los 5 mejores ensayos con esa misma topología.  Si el json ya existe no se pisa (los dos jobs B comparten la arquitectura);
--forzar lo reescribe.   python arq_desde_A.py [--forzar]"""
import os, sys, json
import numpy as np
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
optuna.logging.set_verbosity(optuna.logging.WARNING)

STUDY = os.environ.get("STUDY_NAME_A", "etapa_A_Ext_Clip_Comb_V21")
LOG = os.environ.get("JOURNAL_A", "estudios/optuna_etapa_A_comb.log")
OUT = os.environ.get("ARQ_JSON", os.path.expanduser("~/lightning/estudios/arq_comb_A.json"))
MIN_COMPLETOS = int(os.environ.get("MIN_COMPLETOS_A", 5))

if os.path.exists(OUT) and "--forzar" not in sys.argv:
    try:
        print(f"{OUT} ya existe: {json.load(open(OUT))['arq']}"); sys.exit(0)
    except (ValueError, KeyError):
        print(f"{OUT} existe pero no se pudo leer (¿lo está escribiendo el otro job?): se recalcula", file=sys.stderr)
if not os.path.exists(LOG):
    print(f"ERROR: no existe el journal de la Etapa A ({LOG})", file=sys.stderr); sys.exit(1)
s = optuna.load_study(study_name=STUDY, storage=JournalStorage(JournalFileBackend(LOG)))
ok = sorted([t for t in s.get_trials(deepcopy=False) if t.state.name == "COMPLETE" and t.value is not None
             and np.isfinite(t.value)], key=lambda t: t.value)
if len(ok) < MIN_COMPLETOS:
    print(f"ERROR: la Etapa A tiene solo {len(ok)} ensayos completados (mínimo {MIN_COMPLETOS}).", file=sys.stderr); sys.exit(1)

def topo(t):
    p = t.params; n = p["n_niveles"]
    return (n, p["n_filtros_base"], tuple(p[f"kernel_{i}"] for i in range(n)))

mejor = ok[0]; T = topo(mejor)
grupo = [t for t in ok if topo(t) == T][:5] if "--promedio5" in sys.argv else [mejor]
n = T[0]
drop = [round(float(np.mean([t.params[f"dropout_{i}"] for t in grupo])), 2) for i in range(n)]
arq = dict(n_niveles=n, n_filtros_base=T[1], kernel_sizes=list(T[2]), dropout_ps=drop)
info = dict(arq=arq, mejor_trial=mejor.number, mejor_val=mejor.value, trials_promediados=[t.number for t in grupo],
            dropout_del_mejor=[mejor.params[f"dropout_{i}"] for i in range(n)], completados=len(ok))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
tmp = f"{OUT}.tmp.{os.getpid()}"
json.dump(info, open(tmp, "w"), indent=2)
os.replace(tmp, OUT)          # atómico: los dos jobs B arrancan a la vez y comparten este archivo
print(json.dumps(info, indent=2))
