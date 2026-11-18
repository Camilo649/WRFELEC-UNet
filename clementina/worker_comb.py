"""Un proceso de Optuna por tile. Uso: ETAPA=A|B JOB_INDEX=0|1 python worker_comb.py <i>
Etapa A: TPE con N_STARTUP (10) trials aleatorios, multivariate=False, MedianPruner (15 épocas de calentamiento).
Etapa B: TPE con N_STARTUP (16), multivariate=True, constant_liar=True, MedianPruner (25 épocas de calentamiento);
semilla distinta por proceso."""
import os, sys, warnings
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

idx = int(sys.argv[1])
ETAPA = os.environ.get("ETAPA", "B").upper()
job = int(os.environ.get("JOB_INDEX", 0))
SEED = 42 + 1000 * job + idx
from objetivo_comb import objective          # noqa: E402  (lee ETAPA y, en B, arq_comb_A.json)

STUDY = os.environ.get("STUDY_NAME", f"etapa_{ETAPA}_Ext_Clip_Comb_V21")
LOG = os.environ.get("JOURNAL", f"estudios/optuna_etapa_{ETAPA}_comb.log")
TIMEOUT_S = float(os.environ.get("TIMEOUT_S", 46 * 3600))
# Tope de ensayos registrados, por etapa: el máximo de las dos optimizaciones de la FACL (Temp Ext Clip: A 134, B 108)
MAX_TRIALS = int(os.environ.get("MAX_TRIALS", 134 if ETAPA == "A" else 108))
N_STARTUP = int(os.environ.get("N_STARTUP", 10 if ETAPA == "A" else 16))
os.makedirs(os.path.dirname(LOG) or ".", exist_ok=True)

if ETAPA == "A":
    sampler = optuna.samplers.TPESampler(n_startup_trials=N_STARTUP, multivariate=False, seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=N_STARTUP, n_warmup_steps=15)  # V2: 7; la pérdida de la V2.2 es volátil y se podaban demasiados (bitácora: 15)
else:
    sampler = optuna.samplers.TPESampler(n_startup_trials=N_STARTUP, multivariate=True, constant_liar=True, seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=N_STARTUP, n_warmup_steps=25)  # como la Etapa B de la V2.2

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


class TopeDeEnsayos:
    """Detiene este worker cuando el estudio ya tiene MAX_TRIALS ensayos registrados (cualquier estado), para que la
    cantidad de ensayos sea comparable con la de las optimizaciones de la FACL. Hay un sobrepaso posible de hasta
    (workers - 1) ensayos que ya habían arrancado. MAX_TRIALS=0: sin tope."""
    def __init__(self, n):
        self.n = n

    def __call__(self, study, trial):
        if self.n and len(study.get_trials(deepcopy=False)) >= self.n:
            print(f"[worker] el estudio tiene >= {self.n} ensayos registrados: se detiene este worker.", flush=True)
            study.stop()


class CortarSiFallaSeguido:
    """Detiene ESTE worker si sus últimos 3 trials terminaron en FAIL (los podados no cuentan), como worker_B_temporal.py."""
    def __init__(self, n=3):
        self.n, self.seguidos = n, 0

    def __call__(self, study, trial):
        if trial.state == optuna.trial.TrialState.FAIL:
            self.seguidos += 1
            if self.seguidos >= self.n:
                print(f"[worker] {self.seguidos} trials seguidos en FAIL: se detiene este worker.", flush=True)
                study.stop()
        else:
            self.seguidos = 0


study = optuna.create_study(study_name=STUDY, storage=JournalStorage(JournalFileBackend(LOG)), direction="minimize",
                            sampler=sampler, pruner=pruner, load_if_exists=True)
print(f"[worker_comb {ETAPA} j{job} w{idx}] estudio {STUDY} | journal {LOG} | N_STARTUP {N_STARTUP} | timeout {TIMEOUT_S/3600:.1f} h | tope {MAX_TRIALS} ensayos", flush=True)
study.optimize(objective, timeout=TIMEOUT_S, gc_after_trial=True, catch=(Exception,), callbacks=[CortarSiFallaSeguido(), TopeDeEnsayos(MAX_TRIALS)])
print(f"[worker_comb {ETAPA} j{job} w{idx}] terminó", flush=True)
