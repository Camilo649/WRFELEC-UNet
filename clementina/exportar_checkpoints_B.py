"""Exporta checkpoints de la Etapa B al formato que espera el notebook de entrenamiento
(best_model_state_dict, historial_*, mejor_epoca, ...), un .pth por trial.

  python exportar_checkpoints_B.py --top 5                  # top 5 por mínimo de val Y top 5 por val final
  python exportar_checkpoints_B.py --top 5 --criterio value # solo por mínimo del val_loss
  python exportar_checkpoints_B.py --trials 8 12 27         # trials puntuales

Cada .pth trae:
  best_model_state_dict  pesos de la mejor época (la de mejor val_loss)
  model_state_dict       pesos de la última época
Salida: export_B/B_trialNNNN_<nombre>.pth + export_B/resumen_B.csv
"""
import os, csv, argparse
import torch
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

optuna.logging.set_verbosity(optuna.logging.WARNING)

ap = argparse.ArgumentParser()
ap.add_argument("--top", type=int, default=5)
ap.add_argument("--criterio", choices=["value", "val_final", "ambos"], default="ambos")
ap.add_argument("--trials", type=int, nargs="*", default=None)
ap.add_argument("--nombre", default="Fat Short Unet FACL", help="NOMBRE_MODELO del notebook")
ap.add_argument("--salida", default="export_B")
args = ap.parse_args()

STUDY = os.environ.get("STUDY_NAME", "etapa_B_Fat_Short_Unet_FACL")
LOG   = os.environ.get("JOURNAL", "estudios/optuna_etapa_B.log")
DIR_B = os.environ.get("DIR_B", os.path.expanduser("~/lightning/etapa_B"))

s = optuna.load_study(study_name=STUDY, storage=JournalStorage(JournalFileBackend(LOG)))
ok = [t for t in s.get_trials(deepcopy=False) if t.state.name == "COMPLETE"]
print(f"{len(ok)} trials COMPLETE en el estudio")

if args.trials:
    elegidos = [t for t in ok if t.number in args.trials]
    faltan = set(args.trials) - {t.number for t in elegidos}
    if faltan:
        print("AVISO: no están COMPLETE:", sorted(faltan))
else:
    sel = []
    if args.criterio in ("value", "ambos"):
        sel += sorted(ok, key=lambda t: t.value)[:args.top]
    if args.criterio in ("val_final", "ambos"):
        con_final = [t for t in ok if t.user_attrs.get("val_final") is not None]
        sel += sorted(con_final, key=lambda t: t.user_attrs["val_final"])[:args.top]
    vistos, elegidos = set(), []
    for t in sel:
        if t.number not in vistos:
            vistos.add(t.number)
            elegidos.append(t)

os.makedirs(args.salida, exist_ok=True)
nan = float("nan")
filas = []
for t in elegidos:
    carpeta = t.user_attrs.get("carpeta") or os.path.join(DIR_B, f"trial_{int(t.user_attrs.get('dir_id', t.number)):04d}")
    r_best, r_last, r_curva = (os.path.join(carpeta, f) for f in ("best.pt", "last.pt", "curva.csv"))
    if not all(os.path.exists(r) for r in (r_best, r_last, r_curva)):
        print(f"trial {t.number}: faltan archivos en {carpeta}, se saltea")
        continue

    best = torch.load(r_best, map_location="cpu", weights_only=False)
    last = torch.load(r_last, map_location="cpu", weights_only=False)
    rows = list(csv.DictReader(open(r_curva)))
    col = lambda k: [float(r[k]) for r in rows]
    n = len(rows)

    ckpt = {
        "model_state_dict": last["modelo"],
        "best_model_state_dict": best["modelo"],
        "historial_train_loss": col("train_loss"),
        "historial_val_loss": col("val_loss"),
        "historial_tiempo_epoch": col("seg_epoca"),
        "historial_train_fal": [nan] * n,       # la Etapa B no registró FAL/FCL de entrenamiento por separado
        "historial_train_fcl": [nan] * n,
        "historial_val_fal": col("val_fal"),
        "historial_val_fcl": col("val_fcl"),
        "historial_lr": col("lr"),
        "historial_p_fcl": col("p_fcl"),
        "mejor_epoca": int(best["epoca"]),
        "mejor_val_loss": float(best["val"]),
        "epochs_entrenadas": n,
        "lr": t.params["lr"],
        "patience": "sin early stopping",
        # extras de la Etapa B
        "trial": t.number, "alpha": t.params["alpha"], "epocas": t.params["epocas"],
        "batch_size": t.params["batch_size"],
        "n_parametros": t.user_attrs.get("n_parametros"),
    }
    ruta = os.path.join(args.salida, f"B_trial{t.number:04d}_{args.nombre}.pth")
    torch.save(ckpt, ruta)
    print(f"trial {t.number:4d} | min val {t.value:.5f} (ép {ckpt['mejor_epoca']}) | final {rows[-1]['val_loss'][:7]} | "
          f"{os.path.getsize(ruta)/1e6:.0f} MB -> {ruta}")
    ua = t.user_attrs
    filas.append(dict(trial=t.number, min_val=t.value, mejor_epoca=ckpt["mejor_epoca"],
                      val_final=ua.get("val_final"), fal_mejor=ua.get("val_fal_mejor"), fcl_mejor=ua.get("val_fcl_mejor"),
                      fal_final=ua.get("val_fal_final"), fcl_final=ua.get("val_fcl_final"),
                      lr=t.params["lr"], alpha=t.params["alpha"], epocas=t.params["epocas"],
                      batch_size=t.params["batch_size"], horas=(ua.get("tiempo_trial_seg") or 0) / 3600))

if filas:
    with open(os.path.join(args.salida, "resumen_B.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"\n{len(filas)} checkpoints exportados en {args.salida}/ (más resumen_B.csv)")
