"""Worker de la Etapa B. Uso: python worker_B.py <id_worker_dentro_del_job>
Variables: JOB_INDEX (0 o 1: distingue los dos jobs en paralelo), STUDY_NAME, JOURNAL,
           TIMEOUT_S, N_STARTUP, DIR_B, SMOKE
"""
import os, sys, warnings
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
from objetivo_b import objective_entrenamiento, SEED

worker_id = int(sys.argv[1])
JOB       = int(os.environ.get("JOB_INDEX", 0))
STUDY     = os.environ.get("STUDY_NAME", "etapa_B_Fat_Short_Unet_FACL")
LOG       = os.environ.get("JOURNAL", "estudios/optuna_etapa_B.log")
TIMEOUT_S = float(os.environ.get("TIMEOUT_S", 46 * 3600))
N_STARTUP = int(os.environ.get("N_STARTUP", 24))
os.makedirs(os.path.dirname(LOG) or ".", exist_ok=True)

study = optuna.create_study(
    study_name=STUDY, storage=JournalStorage(JournalFileBackend(LOG)), direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED + worker_id + 1000 * JOB, constant_liar=True,
                                       multivariate=True, n_startup_trials=N_STARTUP),
    pruner=optuna.pruners.NopPruner(),     # cronogramas distintos por trial: no son comparables
    load_if_exists=True,
)
print(f"[worker {worker_id} | job {JOB}] estudio '{STUDY}', log '{LOG}'", flush=True)
study.optimize(objective_entrenamiento, timeout=TIMEOUT_S, catch=(Exception,))
