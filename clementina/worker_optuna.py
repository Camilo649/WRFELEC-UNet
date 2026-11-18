import os, sys, random
import numpy as np, torch, optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from objetivo import (objective_arquitectura, WARMUP_PRUNER_A, SEED, NOMBRE_MODELO)

worker_id = int(sys.argv[1])
STUDY  = os.environ.get("STUDY_NAME", f"busqueda_arquitectura_{NOMBRE_MODELO}")
LOG    = os.environ.get("JOURNAL", "estudios/optuna_arquitectura.log")
TIMEOUT_S = float(os.environ.get("TIMEOUT_S", 46*3600))
os.makedirs(os.path.dirname(LOG), exist_ok=True)

storage = JournalStorage(JournalFileBackend(LOG))
study = optuna.create_study(
    study_name=STUDY, storage=storage, direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED + worker_id, constant_liar=True),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=WARMUP_PRUNER_A),
    load_if_exists=True,
)
study.optimize(objective_arquitectura, timeout=TIMEOUT_S, catch=(Exception,))
