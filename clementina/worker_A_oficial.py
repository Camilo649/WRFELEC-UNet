"""Worker de Optuna para la Etapa A con la FACL oficial (ver objetivo_A_oficial.py).

Lo lanza optuna_A_oficial.sbatch (un worker por tile). Necesita en el entorno N_WORKERS y N_STARTUP:
el n_startup_trials del sampler y el del pruner son el MISMO valor, igual a 2 x cantidad de workers.
Uso:  python worker_A_oficial.py <worker_id>
"""
import os, sys
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

FALLAS_SEGUIDAS_MAX = 3


class CortarSiFallaSeguido:
    """Detiene ESTE worker (study.stop() solo afecta al proceso actual) si sus últimos `n` trials terminaron en
    FAIL (error inesperado): un error sistemático no quema horas lanzando trials que fallan al instante.
    Los trials podados (OOM, NaN, MedianPruner) no cuentan."""
    def __init__(self, n=FALLAS_SEGUIDAS_MAX):
        self.n, self.seguidos = n, 0

    def __call__(self, study, trial):
        if trial.state == optuna.trial.TrialState.FAIL:
            self.seguidos += 1
            if self.seguidos >= self.n:
                print(f"[worker] {self.seguidos} trials seguidos en FAIL: se detiene este worker. "
                      f"Revisá el error de arriba.", flush=True)
                study.stop()
        else:
            self.seguidos = 0


def main(worker_id):
    from objetivo_A_oficial import (objective_arquitectura, WARMUP_PRUNER_A, SEED, NOMBRE_MODELO, DATASET_DIR,
                                    N_CANALES, device, LR_OFICIAL, ETA_MIN_OFICIAL, ALPHA_A, MICRO_BATCH,
                                    EPOCAS_TRIAL_A, BATCH_SIZE_A)

    STUDY     = os.environ.get("STUDY_NAME", f"busqueda_arquitectura_{NOMBRE_MODELO}")
    LOG       = os.environ.get("JOURNAL", "estudios/optuna_arquitectura_oficial.log")   # journal propio
    TIMEOUT_S = float(os.environ.get("TIMEOUT_S", 46*3600))

    try:
        N_WORKERS = int(os.environ["N_WORKERS"])
        N_STARTUP = int(os.environ["N_STARTUP"])
    except KeyError as e:
        raise SystemExit(f"Falta la variable de entorno {e}: este worker se lanza con optuna_A_oficial.sbatch")
    if N_STARTUP != 2 * N_WORKERS:
        raise SystemExit(f"N_STARTUP={N_STARTUP} tiene que ser 2 x N_WORKERS={2 * N_WORKERS}")

    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    storage = JournalStorage(JournalFileBackend(LOG))
    sampler = optuna.samplers.TPESampler(seed=SEED + worker_id, constant_liar=True, multivariate=True,
                                         n_startup_trials=N_STARTUP)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=N_STARTUP, n_warmup_steps=WARMUP_PRUNER_A)
    study = optuna.create_study(
        study_name=STUDY, storage=storage, direction="minimize",
        sampler=sampler, pruner=pruner, load_if_exists=True,
    )
    print(f"[worker {worker_id}] estudio='{STUDY}' journal={LOG} | dataset={DATASET_DIR} canales={N_CANALES} "
          f"| device={device} | TPE multivariate={getattr(sampler, '_multivariate', '?')} "
          f"constant_liar={getattr(sampler, '_constant_liar', '?')} "
          f"n_startup_trials={getattr(sampler, '_n_startup_trials', '?')} "
          f"| MedianPruner n_startup_trials={getattr(pruner, '_n_startup_trials', '?')} "
          f"n_warmup_steps={getattr(pruner, '_n_warmup_steps', '?')} | workers={N_WORKERS} "
          f"| FACL oficial: AdamW lr={LR_OFICIAL:g}{'' if LR_OFICIAL == 1e-3 else ' (PRUEBA DE DIAGNÓSTICO: no es el lr oficial)'}, "
          f"cosine por paso eta_min={ETA_MIN_OFICIAL:g}, "
          f"alpha={ALPHA_A}, micro_batch={MICRO_BATCH}, batch={BATCH_SIZE_A}, épocas={EPOCAS_TRIAL_A}, val=MAE",
          flush=True)
    study.optimize(objective_arquitectura, timeout=TIMEOUT_S, catch=(Exception,),
                   callbacks=[CortarSiFallaSeguido()])


if __name__ == "__main__":
    main(int(sys.argv[1]))
