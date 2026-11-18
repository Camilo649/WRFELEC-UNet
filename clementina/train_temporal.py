"""Entrenamiento del modelo TEMPORAL (9 canales: vars en t | t-(t-1) | (t-1)-(t-2)) con una
configuración fija de la Etapa B. Misma arquitectura que el baseline; solo cambian in_channels y el dataset.

Uso (desde ~/lightning, con DATASET_DIR apuntando al dataset temporal):
    python train_temporal.py --trial 39                 # lr/alpha/epocas/batch_size leídos del estudio de Optuna
    python train_temporal.py --lr 1e-4 --alpha 0.1 --epocas 100 --batch-size 16
    python train_temporal.py --trial 39 --epocas 2      # los argumentos explícitos pisan a los del trial (smoke test)

Guarda SOLO dos checkpoints (se sobrescriben a medida que avanza el entrenamiento):
    checkpoints/temporal_<RUN>_best.pth   mejor época según val_loss
    checkpoints/temporal_<RUN>_last.pth   última época terminada (+ optimizer/scheduler)
Ambos tienen las claves que usa el notebook de resultados ('model_state_dict', 'historial_*', 'mejor_epoca', ...).
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from modelo import UNet_WRF, FACL
from datos import train_dataset, val_dataset, DATASET_DIR

SEED = 42
device = torch.device(os.environ.get("DEVICE", "xpu"))     # DEVICE=cpu solo para smoke tests chicos

# --- Arquitectura final (idéntica al baseline) ---
N_NIVELES      = 3
N_FILTROS_BASE = 64
KERNEL_SIZES   = [5, 5, 3]
DROPOUT_PS     = [0.03, 0.08, 0.26]

# --- Dónde está el estudio de Etapa B (mismos defaults que estado_B.py) ---
STUDY       = os.environ.get("STUDY_NAME", "etapa_B_Ext_Clip_U-Net_FACL")
JOURNAL     = os.environ.get("JOURNAL", "estudios/optuna_etapa_B.log")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 0))     # 0 = como en train_best_f128.py

RUN = os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d_%H%M"))
CARPETA = "checkpoints"
RUTA_BEST = os.path.join(CARPETA, f"temporal_{RUN}_best.pth")
RUTA_LAST = os.path.join(CARPETA, f"temporal_{RUN}_last.pth")


class FACLConTermino(FACL):
    """Idéntica a FACL (mismo sorteo: un np.random.rand() por paso, misma regla p > P(t) -> FAL),
    pero además registra qué término se usó en cada paso. No se toca modelo.py."""
    ultimo_termino = None

    def forward(self, pred, gt, mode='train'):
        if mode == 'val':
            return super().forward(pred, gt, mode)
        fft_pred, fft_gt = self._fft(pred), self._fft(gt)
        if np.random.rand() > self.prob_fcl():
            loss, self.ultimo_termino = self.fal(fft_pred, fft_gt), 'FAL'
        else:
            loss, self.ultimo_termino = self.fcl(fft_pred, fft_gt), 'FCL'
        self.step += 1
        return loss


def resolver_hiperparametros(args):
    hp = {}
    if args.trial is not None:
        import optuna
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        estudio = optuna.load_study(study_name=STUDY, storage=JournalStorage(JournalFileBackend(JOURNAL)))
        trials = [t for t in estudio.get_trials(deepcopy=False) if t.number == args.trial]
        if not trials:
            raise SystemExit(f"No existe el trial #{args.trial} en el estudio '{STUDY}' ({JOURNAL})")
        t = trials[0]
        if t.state.name != "COMPLETE":
            print(f"[aviso] el trial #{t.number} está en estado {t.state.name}")
        hp = {k: t.params[k] for k in ("lr", "alpha", "epocas", "batch_size")}
        print(f"Hiperparámetros del trial #{t.number} (val_loss={t.value})")
    for k, v in (("lr", args.lr), ("alpha", args.alpha), ("epocas", args.epocas), ("batch_size", args.batch_size)):
        if v is not None:
            hp[k] = v
    faltan = [k for k in ("lr", "alpha", "epocas", "batch_size") if k not in hp]
    if faltan:
        raise SystemExit(f"Faltan hiperparámetros: {faltan}. Usá --trial N o pasalos con --lr --alpha --epocas --batch-size")
    return {"lr": float(hp["lr"]), "alpha": float(hp["alpha"]),
            "epocas": int(hp["epocas"]), "batch_size": int(hp["batch_size"])}


def guardar(estado, ruta):
    """Escritura atómica: si el job se corta a mitad de un guardado, el checkpoint anterior queda intacto."""
    tmp = ruta + ".tmp"
    torch.save(estado, tmp)
    os.replace(tmp, ruta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--epocas", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    hp = resolver_hiperparametros(ap.parse_args())
    LR, ALPHA, EPOCAS, BATCH_SIZE = hp["lr"], hp["alpha"], hp["epocas"], hp["batch_size"]

    # Chequeo de canales: el dataset temporal tiene que tener 9 (si DATASET_DIR apunta al original, hay 3)
    stats = np.load(os.path.join(DATASET_DIR, "stats.npz"))
    N_CANALES = int(stats["min_X"].shape[0])
    nombres_canales = [str(n) for n in stats["canales"]] if "canales" in stats.files else None
    assert N_CANALES == 9, (f"stats.npz de {DATASET_DIR} tiene {N_CANALES} canales (se esperaban 9): "
                            f"¿DATASET_DIR apunta al dataset original?")

    print(f"Dataset: {DATASET_DIR} | canales: {N_CANALES} {nombres_canales}")
    print(f"Hiperparámetros: lr={LR:.3e} alpha={ALPHA:.3f} épocas={EPOCAS} batch={BATCH_SIZE} | seed={SEED} | workers={NUM_WORKERS} | device={device}")
    print(f"Checkpoints: {RUTA_BEST} | {RUTA_LAST}", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator,
                              num_workers=NUM_WORKERS, pin_memory=False, persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=False, persistent_workers=NUM_WORKERS > 0)

    modelo = UNet_WRF(in_channels=N_CANALES, n_niveles=N_NIVELES, n_filtros_base=N_FILTROS_BASE,
                      kernel_sizes=KERNEL_SIZES, dropout_ps=DROPOUT_PS).to(device)
    print(f"Arquitectura: {N_NIVELES} niveles, {N_FILTROS_BASE} filtros, kernels {KERNEL_SIZES}, dropouts {DROPOUT_PS}")
    print(f"Parámetros: {sum(p.numel() for p in modelo.parameters()):,}", flush=True)

    criterion = FACLConTermino(total_steps=EPOCAS * len(train_loader), alpha=ALPHA).to(device)
    optimizer = optim.Adam(modelo.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCAS, eta_min=1e-7)

    h = {k: [] for k in ("train_loss", "val_loss", "train_fal", "train_fcl", "val_fal", "val_fcl", "lr", "tiempo_epoch")}
    mejor_val_loss, mejor_epoca = float("inf"), 0
    os.makedirs(CARPETA, exist_ok=True)

    def armar_checkpoint(epoca_actual):
        return {
            "model_state_dict": {k: v.detach().cpu() for k, v in modelo.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "historial_train_loss": list(h["train_loss"]), "historial_val_loss": list(h["val_loss"]),
            "historial_train_fal": list(h["train_fal"]), "historial_train_fcl": list(h["train_fcl"]),
            "historial_val_fal": list(h["val_fal"]), "historial_val_fcl": list(h["val_fcl"]),
            "historial_lr": list(h["lr"]), "historial_tiempo_epoch": list(h["tiempo_epoch"]),
            "mejor_epoca": mejor_epoca, "mejor_val_loss": mejor_val_loss,
            "epochs_entrenadas": epoca_actual,
            "lr": LR, "alpha": ALPHA, "epocas": EPOCAS, "batch_size": BATCH_SIZE, "patience": None,
            "criterion_step": criterion.step, "seed": SEED, "run": RUN,
            "nombres_canales": nombres_canales, "in_channels": N_CANALES,
            "arquitectura": {"n_niveles": N_NIVELES, "n_filtros_base": N_FILTROS_BASE,
                             "kernel_sizes": KERNEL_SIZES, "dropout_ps": DROPOUT_PS},
        }

    t_inicio = time.time()
    for epoch in range(EPOCAS):
        t0 = time.time()
        lr_epoca = optimizer.param_groups[0]["lr"]

        # ---------- Train ----------
        modelo.train()
        acc_loss = 0.0
        suma_fal = suma_fcl = 0.0
        n_fal = n_fcl = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(modelo(batch_X), batch_y, mode="train")
            loss.backward()
            optimizer.step()
            loss_v = loss.item()
            acc_loss += loss_v * batch_X.size(0)
            if criterion.ultimo_termino == "FAL":
                suma_fal += loss_v; n_fal += 1
            else:
                suma_fcl += loss_v; n_fcl += 1
        train_loss = acc_loss / len(train_loader.dataset)

        # ---------- Val (sobre todo el set, acumulando sumas) ----------
        modelo.eval()
        criterion.val_reset()
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                criterion.val_acumular(modelo(batch_X), batch_y)
        val_loss = criterion.val_resultado()
        scheduler.step()

        # ---------- Historiales ----------
        h["train_loss"].append(train_loss); h["val_loss"].append(val_loss)
        h["train_fal"].append(suma_fal / n_fal if n_fal else float("nan"))
        h["train_fcl"].append(suma_fcl / n_fcl if n_fcl else float("nan"))
        h["val_fal"].append(criterion.val_fal); h["val_fcl"].append(criterion.val_fcl)
        h["lr"].append(lr_epoca); h["tiempo_epoch"].append(time.time() - t0)

        # ---------- Checkpoints: mejor época y última ----------
        es_mejor = val_loss < mejor_val_loss
        if es_mejor:
            mejor_val_loss, mejor_epoca = val_loss, epoch + 1
        ck = armar_checkpoint(epoch + 1)
        if es_mejor:
            guardar({k: v for k, v in ck.items() if k not in ("optimizer_state_dict", "scheduler_state_dict")}, RUTA_BEST)
        guardar(ck, RUTA_LAST)

        print(f"Epoch {epoch+1:03d}/{EPOCAS} | {h['tiempo_epoch'][-1]:.0f}s | LR {lr_epoca:.2e} | "
              f"Train {train_loss:.5f} | Val {val_loss:.5f} (FAL {criterion.val_fal:.5f} FCL {criterion.val_fcl:.4f})"
              f"{'  <- mejor' if es_mejor else ''}", flush=True)

    print(f"\nEntrenamiento finalizado en {(time.time() - t_inicio) / 3600:.2f} horas.")
    print(f"Mejor época: {mejor_epoca} (val_loss {mejor_val_loss:.6f}) -> {RUTA_BEST}")
    print(f"Última época: {EPOCAS} (val_loss {h['val_loss'][-1]:.6f}) -> {RUTA_LAST}")


if __name__ == "__main__":
    main()
