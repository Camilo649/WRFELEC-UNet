import os, sys, random
import numpy as np, torch, optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from objetivo_A_temporal import (objective_arquitectura, WARMUP_PRUNER_A, SEED, NOMBRE_MODELO,
                                 DATASET_DIR, N_CANALES)

worker_id = int(sys.argv[1])
STUDY  = os.environ.get("STUDY_NAME", f"busqueda_arquitectura_{NOMBRE_MODELO}")
LOG    = os.environ.get("JOURNAL", "estudios/optuna_arquitectura_temporal.log")      # journal NUEVO: no se mezcla con el de 3 canales
TIMEOUT_S = float(os.environ.get("TIMEOUT_S", 46*3600))
MULTIVARIATE = True                                                                  # TPE multivariado
N_STARTUP_TRIALS = 16                                                                # trials aleatorios antes de que arranque el TPE
os.makedirs(os.path.dirname(LOG), exist_ok=True)

storage = JournalStorage(JournalFileBackend(LOG))
sampler = optuna.samplers.TPESampler(seed=SEED + worker_id, constant_liar=True, multivariate=MULTIVARIATE,
                                     n_startup_trials=N_STARTUP_TRIALS)
study = optuna.create_study(
    study_name=STUDY, storage=storage, direction="minimize",
    sampler=sampler,
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=WARMUP_PRUNER_A),
    load_if_exists=True,
)
print(f"[worker {worker_id}] estudio='{STUDY}' journal={LOG} | dataset={DATASET_DIR} canales={N_CANALES} "
      f"| TPE multivariate={sampler._multivariate} constant_liar={sampler._constant_liar} "
      f"n_startup_trials={sampler._n_startup_trials}", flush=True)
study.optimize(objective_arquitectura, timeout=TIMEOUT_S, catch=(Exception,))
