"""Reentrena la configuración (lr, alpha, batch_size) del mejor trial de la Etapa B OFICIAL con otra cantidad
de épocas, sin tocar el estudio original ni la carpeta etapa_B_oficial/.

Uso:  python run_mejor_200.py          (se configura con variables de entorno; ver abajo)

Variables (todas opcionales):
  TRIAL_BASE      trial del estudio original del que se copian lr, alpha y batch_size   (100)
  EPOCAS          épocas del reentrenamiento; tiene que ser una de las opciones del espacio (20..200)  (200)
  DIR_ID          carpeta trial_XXXX y semilla (SEED + DIR_ID); con DIR_ID = TRIAL_BASE se usa la MISMA
                  semilla que el trial original (misma inicialización y mismo orden de lotes)       (TRIAL_BASE)
  JOURNAL_ORIGEN  journal del estudio original           (estudios/optuna_etapa_B_oficial.log)
  JOURNAL_NUEVO   journal de esta corrida                (estudios/optuna_mejor_oficial_<EPOCAS>ep_id<DIR_ID>.log)
  DIR_B           carpeta de salida                      (~/lightning/etapa_B_oficial_<EPOCAS>)

Salida: DIR_B/trial_<DIR_ID>/{best,best_csi,best_fss,last}.pt, curva.csv y resume.pt (igual que la Etapa B).
Si Slurm corta el job, se vuelve a lanzar con el mismo comando y se reanuda desde resume.pt.
"""
import os
import sys
import warnings

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

TRIAL_BASE = int(os.environ.get("TRIAL_BASE", 100))
EPOCAS = int(os.environ.get("EPOCAS", 200))
DIR_ID = int(os.environ.get("DIR_ID", TRIAL_BASE))
JOURNAL_ORIGEN = os.environ.get("JOURNAL_ORIGEN", "estudios/optuna_etapa_B_oficial.log")
JOURNAL_NUEVO = os.environ.get("JOURNAL_NUEVO", f"estudios/optuna_mejor_oficial_{EPOCAS}ep_id{DIR_ID}.log")
ESTUDIO_ORIGEN = "etapa_B_Temp_Ext_Clip_U-Net_FACL_oficial"
ESTUDIO_NUEVO = f"mejor_oficial_t{TRIAL_BASE}_{EPOCAS}ep_id{DIR_ID}"
# La carpeta se fija ANTES de importar el objetivo (lo lee al importarse)
os.environ.setdefault("DIR_B", os.path.expanduser(f"~/lightning/etapa_B_oficial_{EPOCAS}"))


def main():
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    from objetivo_B_oficial import objective_entrenamiento, EPOCAS_OPCIONES, DIR_B, device

    assert EPOCAS in EPOCAS_OPCIONES, f"EPOCAS={EPOCAS} no está en {EPOCAS_OPCIONES}"
    if os.path.abspath(DIR_B).rstrip("/").endswith("etapa_B_oficial"):
        raise SystemExit("DIR_B apunta a la carpeta de la Etapa B original: no se pisa.")

    # lr, alpha y batch_size se leen del estudio original (no se copian a mano)
    origen = optuna.load_study(study_name=ESTUDIO_ORIGEN, storage=JournalStorage(JournalFileBackend(JOURNAL_ORIGEN)))
    t0 = next(t for t in origen.trials if t.number == TRIAL_BASE)
    assert t0.state == optuna.trial.TrialState.COMPLETE, f"el trial {TRIAL_BASE} no está COMPLETE ({t0.state})"
    params = dict(t0.params)
    params["epocas"] = EPOCAS
    print(f"[run_mejor_200] trial base {TRIAL_BASE}: {t0.params} (val {t0.value:.6f}) -> reentrenar con {params} | "
          f"DIR_ID={DIR_ID} (semilla 42+{DIR_ID}) | device={device} | carpeta={DIR_B}", flush=True)

    os.makedirs(os.path.dirname(JOURNAL_NUEVO) or ".", exist_ok=True)
    study = optuna.create_study(study_name=ESTUDIO_NUEVO, storage=JournalStorage(JournalFileBackend(JOURNAL_NUEVO)),
                                direction="minimize", pruner=optuna.pruners.NopPruner(), load_if_exists=True)
    if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        print("[run_mejor_200] esta corrida ya terminó (hay un trial COMPLETE en el journal); no hago nada.", flush=True)
        return
    # Si hay un trial anterior cortado (RUNNING/FAIL), se encola otro con la misma carpeta: reanuda desde resume.pt
    study.enqueue_trial(params, user_attrs={"resume_from": DIR_ID})
    study.optimize(objective_entrenamiento, n_trials=1, catch=())     # catch=(): que un error se vea, no se trague


if __name__ == "__main__":
    sys.exit(main())
