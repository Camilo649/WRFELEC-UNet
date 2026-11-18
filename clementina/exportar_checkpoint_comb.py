"""Exporta ensayos de la Etapa B de la Ext Clip U-Net Comb V2.1 al formato que espera el notebook
(best_model_state_dict, model_state_dict, historial_*, mejor_epoca, ...), un .pth por ensayo.

  cd ~/lightning && python exportar_checkpoint_comb.py              # el mejor ensayo (mínimo de la validación)
  cd ~/lightning && python exportar_checkpoint_comb.py --trials 92 77

Cada .pth trae:
  best_model_state_dict  pesos de la mejor época (la de menor pérdida de validación híbrida)
  model_state_dict       pesos de la última época
  historial_*            curva época por época (de curva.csv), incluidos WMSE, FAL, FCL, CSI y FSS de validación
Salida: export_B_comb/B_trialNNNN_Ext Clip U-Net Comb V2.1.pth + export_B_comb/resumen_B_comb.csv
"""
import os, csv, argparse
import torch
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

optuna.logging.set_verbosity(optuna.logging.WARNING)

ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, nargs="*", default=None, help="sin argumento: el mejor ensayo")
ap.add_argument("--nombre", default="Ext Clip U-Net Comb V2.1", help="NOMBRE_MODELO del notebook")
ap.add_argument("--salida", default="export_B_comb")
args = ap.parse_args()

STUDY = os.environ.get("STUDY_NAME", "etapa_B_Ext_Clip_Comb_V21")
LOG = os.environ.get("JOURNAL", "estudios/optuna_etapa_B_comb.log")
DIR_B = os.environ.get("DIR_COMB", os.path.expanduser("~/lightning/etapa_B_comb"))
FIJOS = dict(epocas=150, paciencia=15, w_pos=17.53, rampa_epocas=60)      # FIJOS_B de objetivo_comb.py

s = optuna.load_study(study_name=STUDY, storage=JournalStorage(JournalFileBackend(LOG)))
ok = [t for t in s.get_trials(deepcopy=False) if t.state.name == "COMPLETE"]
print(f"{len(ok)} ensayos COMPLETE en el estudio {STUDY}")
if args.trials:
    elegidos = [t for t in ok if t.number in args.trials]
    faltan = set(args.trials) - {t.number for t in elegidos}
    if faltan:
        print("AVISO: no están COMPLETE:", sorted(faltan))
else:
    elegidos = [min(ok, key=lambda t: t.value)]

os.makedirs(args.salida, exist_ok=True)
filas = []
for t in elegidos:
    carpeta = t.user_attrs.get("carpeta") or os.path.join(DIR_B, f"trial_{int(t.user_attrs.get('dir_id', t.number)):04d}")
    r_best, r_last, r_curva = (os.path.join(carpeta, f) for f in ("best.pt", "last.pt", "curva.csv"))
    if not all(os.path.exists(r) for r in (r_best, r_last, r_curva)):
        print(f"ensayo {t.number}: faltan archivos en {carpeta}, se saltea")
        continue
    best = torch.load(r_best, map_location="cpu", weights_only=False)
    last = torch.load(r_last, map_location="cpu", weights_only=False)
    rows = list(csv.DictReader(open(r_curva)))
    col = lambda k: [float(r[k]) for r in rows]
    n = len(rows)
    # Clementina guarda el ensayo con su propio número de carpeta; se verifica que sea el mismo mínimo
    assert abs(float(best["val"]) - t.value) < 1e-6, (best["val"], t.value)

    ckpt = {
        "model_state_dict": last["modelo"],
        "best_model_state_dict": best["modelo"],
        "historial_train_loss": col("train_loss"),
        "historial_val_loss": col("val_loss"),
        "historial_tiempo_epoch": col("seg_epoca"),
        "historial_val_wmse": col("val_wmse"),
        "historial_val_fal": col("val_fal"),          # sin escalar por sqrt(HW)
        "historial_val_fcl": col("val_fcl"),          # sin escalar por sqrt(HW)
        "historial_q_wmse": col("q_wmse"),
        "historial_lr": col("lr"),
        "historial_csi": col("csi_media"),
        "historial_fss_u15_v3": col("fss_u15_v3"),
        "mejor_epoca": int(best["epoca"]),
        "mejor_val_loss": float(best["val"]),
        "epochs_entrenadas": n,
        "lr": t.params["lr"],
        "patience": FIJOS["paciencia"],
        # extras de la Etapa B
        "trial": t.number, "batch_size": t.params["batch_size"], "max_prob": t.params["max_prob"],
        "max_norm": t.params["max_norm"], "w_pos": FIJOS["w_pos"], "rampa_epocas": FIJOS["rampa_epocas"],
        "epocas_max": FIJOS["epocas"], "n_parametros": t.user_attrs.get("n_parametros"),
    }
    ruta = os.path.join(args.salida, f"B_trial{t.number:04d}_{args.nombre}.pth")
    torch.save(ckpt, ruta)
    print(f"ensayo {t.number:4d} | min val {t.value:.5f} (ép {ckpt['mejor_epoca']}) | última {rows[-1]['val_loss'][:8]} "
          f"(ép {n}) | {os.path.getsize(ruta)/1e6:.0f} MB -> {ruta}")
    filas.append(dict(trial=t.number, min_val=t.value, mejor_epoca=ckpt["mejor_epoca"], val_final=float(rows[-1]["val_loss"]),
                      epocas_entrenadas=n, lr=t.params["lr"], batch_size=t.params["batch_size"],
                      max_prob=t.params["max_prob"], max_norm=t.params["max_norm"],
                      horas=(t.user_attrs.get("tiempo_trial_seg") or 0) / 3600))

if filas:
    with open(os.path.join(args.salida, "resumen_B_comb.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"\n{len(filas)} checkpoint(s) exportado(s) en {args.salida}/ (más resumen_B_comb.csv)")
