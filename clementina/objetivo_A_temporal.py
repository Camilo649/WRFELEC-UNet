"""Etapa A (búsqueda de arquitectura) con el dataset TEMPORAL (9 canales).

Es objetivo.py con cuatro cambios, nada más (mismo presupuesto por trial, mismo espacio de búsqueda salvo n_niveles,
misma FACL, mismos lr/alpha/batch_size, para que sea comparable con la Etapa A de 3 canales):
  1. in_channels sale de stats.npz del dataset (y se exige que sean 9).
  2. NOMBRE_MODELO distinto -> estudio distinto.
  3. Se guardan 'dataset' y 'n_canales' como user_attrs de cada trial.
  4. n_niveles fijo en 3 (antes se buscaba entre 3 y 4).
"""
import os, time
import numpy as np
import torch
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader

from modelo import UNet_WRF, FACL
from datos import train_dataset, val_dataset, DATASET_DIR

SEED = 42
NOMBRE_MODELO = 'Temp Ext Clip U-Net FACL'
device = torch.device("xpu")

# ==========================================
# Chequeo de canales: el dataset temporal tiene 9 (si DATASET_DIR apunta al original, hay 3)
# ==========================================
_stats = np.load(os.path.join(DATASET_DIR, "stats.npz"))
N_CANALES = int(_stats["min_X"].shape[0])
assert N_CANALES == 9, (f"stats.npz de {DATASET_DIR} tiene {N_CANALES} canales (se esperaban 9): "
                        f"¿DATASET_DIR apunta al dataset original?")
print(f"[objetivo_A_temporal] Dataset: {DATASET_DIR} | canales: {N_CANALES}", flush=True)

# Se pueden sobreescribir por variable de entorno para la prueba corta

# ==========================================
# ETAPA A — presupuesto FIJO por trial
# ==========================================
# El cronograma de FACL (P(t)) se escala con los pasos totales, así que todos los trials
# recorren la misma "película" (FCL -> FAL) y son comparables entre sí. Sin early stopping.
EPOCAS_TRIAL_A  = int(os.environ.get("EPOCAS_TRIAL_A", 50))
BATCH_SIZE_A    = 16
LR_ETAPA_A      = 2.5e-5
ALPHA_A         = 0.1
N_NIVELES       = 3                                    # profundidad fija: la búsqueda cubre solo filtros base, dropout y kernel por nivel
WARMUP_PRUNER_A = int(float(os.environ.get("WARMUP_FRAC", 0.6)) * EPOCAS_TRIAL_A)       # el pruner recién actúa pasado el 60% del cronograma

def objective_arquitectura(trial):
    tiempo_inicio_trial = time.time()

    # ==========================================
    # HIPERPARÁMETROS DE ARQUITECTURA
    # ==========================================
    n_niveles      = N_NIVELES                     # FIJO en 3 (decisión con la directora): ya no se busca
    n_filtros_base = trial.suggest_categorical('n_filtros_base', [16, 32, 64])

    dropout_ps = [
        trial.suggest_float(f'dropout_nivel_{i}', 0.0, 0.3)
        for i in range(n_niveles)
    ]

    kernel_sizes = [
        trial.suggest_categorical(f'kernel_nivel_{i}', [3, 5, 7, 9])
        for i in range(n_niveles)
    ]

    # ==========================================
    # DATALOADERS
    # ==========================================
    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader_trial = DataLoader(train_dataset, batch_size=BATCH_SIZE_A,
                                     shuffle=True, generator=generator,
                                     num_workers=0, pin_memory=False)
    val_loader_trial   = DataLoader(val_dataset, batch_size=BATCH_SIZE_A,
                                     shuffle=False, num_workers=0, pin_memory=False)

    # ==========================================
    # MODELO, LOSS, OPTIMIZER Y SCHEDULER
    # ==========================================
    torch.manual_seed(SEED + trial.number)
    np.random.seed(SEED + trial.number)          # sorteo FAL/FCL reproducible por trial
    modelo_trial = UNet_WRF(
        in_channels=N_CANALES,                   # 9 (dataset temporal)
        n_niveles=n_niveles, n_filtros_base=n_filtros_base,
        kernel_sizes=kernel_sizes, dropout_ps=dropout_ps
    ).to(device)
    n_parametros = sum(p.numel() for p in modelo_trial.parameters())

    criterion_trial = FACL(total_steps=EPOCAS_TRIAL_A * len(train_loader_trial),
                           alpha=ALPHA_A).to(device)

    optimizer_trial = optim.Adam(modelo_trial.parameters(), lr=LR_ETAPA_A)

    scheduler_trial = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_trial, T_max=EPOCAS_TRIAL_A, eta_min=1e-7
    )

    mejor_objetivo = float('inf')
    mejor_epoca    = 0
    val_fal_mejor  = val_fcl_mejor = None

    # ==========================================
    # BUCLE DE ENTRENAMIENTO DEL TRIAL
    # ==========================================
    try:
        for epoch in range(EPOCAS_TRIAL_A):
            modelo_trial.train()
            for batch_X, batch_y in train_loader_trial:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer_trial.zero_grad()
                pred = modelo_trial(batch_X)
                loss = criterion_trial(pred, batch_y, mode='train')
                loss.backward()
                optimizer_trial.step()

            # Validación sobre TODO el set (FCL global: se acumulan las sumas)
            modelo_trial.eval()
            criterion_trial.val_reset()
            with torch.no_grad():
                for batch_X, batch_y in val_loader_trial:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    criterion_trial.val_acumular(modelo_trial(batch_X), batch_y)
            epoch_val_loss = criterion_trial.val_resultado()

            # Detectar NaN/Inf y podar el trial en lugar de dejarlo completarse con basura
            if not np.isfinite(epoch_val_loss):
                print(f"[Trial {trial.number}] val_loss no finito ({epoch_val_loss}) — podando trial.")
                raise optuna.TrialPruned()

            scheduler_trial.step()

            if epoch_val_loss < mejor_objetivo:
                mejor_objetivo = epoch_val_loss
                mejor_epoca    = epoch + 1
                val_fal_mejor, val_fcl_mejor = criterion_trial.val_fal, criterion_trial.val_fcl

            trial.report(epoch_val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            print(f"[Trial {trial.number}] Epoch {epoch+1}/{EPOCAS_TRIAL_A} | val_loss={epoch_val_loss:.4f} "
                  f"(FAL={criterion_trial.val_fal:.5f} FCL={criterion_trial.val_fcl:.4f})")

        trial.set_user_attr('tiempo_trial_seg', time.time() - tiempo_inicio_trial)
        trial.set_user_attr('n_parametros', n_parametros)
        trial.set_user_attr('mejor_epoca', mejor_epoca)
        trial.set_user_attr('val_final', epoch_val_loss)
        trial.set_user_attr('val_fal_mejor', val_fal_mejor)
        trial.set_user_attr('val_fcl_mejor', val_fcl_mejor)
        trial.set_user_attr('n_canales', N_CANALES)
        trial.set_user_attr('dataset', DATASET_DIR)
        return mejor_objetivo

    except optuna.TrialPruned:
        raise

    except torch.xpu.OutOfMemoryError:
        torch.xpu.empty_cache()
        print(f"[Trial {trial.number}] OOM — se descarta este trial.")
        raise optuna.TrialPruned()

    except Exception as e:
        torch.xpu.empty_cache()
        print(f"[Trial {trial.number}] Error inesperado: {type(e).__name__}: {e} — se descarta este trial.")
        raise optuna.TrialPruned()
