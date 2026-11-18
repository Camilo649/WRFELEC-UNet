"""Worker de la Etapa B temporal. Uso: python worker_B_temporal.py <id_worker_dentro_del_job>
Variables: JOB_INDEX (0 o 1: distingue los dos jobs en paralelo), N_STARTUP (obligatoria; la exporta el sbatch),
           STUDY_NAME, JOURNAL, TIMEOUT_S, DIR_B, SMOKE
Igual que worker_B.py, salvo: estudio y journal propios, N_STARTUP obligatoria, y un worker se detiene solo si
3 trials seguidos terminan en FAIL (error inesperado) para no quemar horas con un error sistemático.
"""
import os, sys, warnings
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

FALLAS_SEGUIDAS_MAX = 3


class CortarSiFallaSeguido:
    """Detiene ESTE worker (study.stop() solo afecta al proceso actual) si sus últimos `n` trials terminaron en
    FAIL. Los trials podados (OOM, NaN) no cuentan."""
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
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    from objetivo_B_temporal import (objective_entrenamiento, SEED, ARQ_FINAL, DIR_B, DATASET_DIR, N_CANALES,
                                     device, LR_RANGO, ALPHA_RANGO, EPOCAS_OPCIONES, BATCH_OPCIONES)

    JOB       = int(os.environ.get("JOB_INDEX", 0))
    STUDY     = os.environ.get("STUDY_NAME", "etapa_B_Temp_Ext_Clip_U-Net_FACL")
    LOG       = os.environ.get("JOURNAL", "estudios/optuna_etapa_B_temporal.log")
    TIMEOUT_S = float(os.environ.get("TIMEOUT_S", 46 * 3600))
    try:
        N_STARTUP = int(os.environ["N_STARTUP"])
    except KeyError:
        raise SystemExit("Falta la variable de entorno N_STARTUP: este worker se lanza con sbatch_etapa_B_temporal.sh")
    os.makedirs(os.path.dirname(LOG) or ".", exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=SEED + worker_id + 1000 * JOB, constant_liar=True,
                                         multivariate=True, n_startup_trials=N_STARTUP)
    study = optuna.create_study(
        study_name=STUDY, storage=JournalStorage(JournalFileBackend(LOG)), direction="minimize",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),     # cronogramas distintos por trial: no son comparables
        load_if_exists=True,
    )
    print(f"[worker {worker_id} | job {JOB}] estudio '{STUDY}', log '{LOG}' | dataset={DATASET_DIR} canales={N_CANALES} "
          f"| device={device} | arq={ARQ_FINAL} | carpeta={DIR_B} | TPE multivariate={getattr(sampler, '_multivariate', '?')} "
          f"constant_liar={getattr(sampler, '_constant_liar', '?')} n_startup_trials={getattr(sampler, '_n_startup_trials', '?')} "
          f"| lr={LR_RANGO} alpha={ALPHA_RANGO} épocas={EPOCAS_OPCIONES} batch={BATCH_OPCIONES}", flush=True)
    study.optimize(objective_entrenamiento, timeout=TIMEOUT_S, catch=(Exception,),
                   callbacks=[CortarSiFallaSeguido()])


if __name__ == "__main__":
    main(int(sys.argv[1]))
