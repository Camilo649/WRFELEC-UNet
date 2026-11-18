"""Etapa A (búsqueda de arquitectura) con el dataset TEMPORAL (9 canales) y la FACL implementada TAL CUAL
el código oficial: https://github.com/argenycw/FACL (commit f46009f), tomando train_meteo.py como referencia.

Igual que objetivo_A_temporal.py (lo único que cambia es la FACL y el entrenamiento): dataset temporal,
espacio de búsqueda (n_niveles = 3 fijo; filtros base, dropout y kernel por nivel), 50 épocas por trial sin
early stopping, batch_size = 16, alpha = 0.1, semillas, MedianPruner con warmup del 60 %.

Lo que cambia, para que coincida con el código oficial en todas las filas de la Tabla 4.1:
  - Pérdida: RandomScheduling oficial copiada textual (facl_oficial.py): un sorteo FAL/FCL por lote
    (micro_batch = 1), FCL con una sola raíz y sin eps, pérdida x sqrt(H*W). get_loss('facl-0.1') = línea 37.
  - Optimizador: torch.optim.AdamW(params, lr=1e-3), resto de argumentos por defecto (train_meteo.py:99).
  - Tasa de aprendizaje: CosineAnnealingLR(T_max = pasos totales, eta_min = 1e-6) con un scheduler.step()
    por cada optimizer.step() (train_meteo.py:116 y 164-168). Pasos totales = len(train_loader) * épocas /
    micro_batch (train_meteo.py:82), el mismo valor que recibe RandomScheduling.
  - Mejor modelo = objetivo de Optuna: el menor MAE de validación. El MAE es el de utilspp.py:302
    (F.l1_loss) promediado por lote como MetricListEvaluator (eval.py:158-178), sobre los valores
    normalizados con los que se entrena.
Validación: en todas las épocas y sobre el set de validación completo. El código oficial no tiene una
convención única (train_meteo.py valida cada 2 épocas, train_sevir.py cada época, ambos sobre 19 lotes);
acá se usa el set completo para que la comparación entre arquitecturas no dependa de qué tormentas caen
en esos lotes.
"""
import os, time, random
import numpy as np
import torch
import optuna
from torch.utils.data import DataLoader

from modelo import UNet_WRF                    # la misma U-Net del primer estudio (no se usa su FACL)
from facl_oficial import RandomScheduling      # copia textual de utilspp.RandomScheduling
from datos import train_dataset, val_dataset, DATASET_DIR

SEED = 42
NOMBRE_MODELO = 'Temp Ext Clip U-Net FACL oficial'
device = torch.device(os.environ.get("DEVICE", "xpu"))     # DEVICE=cpu solo para pruebas locales

# ==========================================
# Chequeo de canales: el dataset temporal tiene 9 (si DATASET_DIR apunta al original, hay 3)
# ==========================================
_stats = np.load(os.path.join(DATASET_DIR, "stats.npz"))
N_CANALES = int(_stats["min_X"].shape[0])
assert N_CANALES == 9, (f"stats.npz de {DATASET_DIR} tiene {N_CANALES} canales (se esperaban 9): "
                        f"¿DATASET_DIR apunta al dataset original?")
print(f"[objetivo_A_oficial] Dataset: {DATASET_DIR} | canales: {N_CANALES} | device: {device}", flush=True)

# ==========================================
# ETAPA A — presupuesto FIJO por trial (igual que el primer estudio)
# ==========================================
EPOCAS_TRIAL_A  = int(os.environ.get("EPOCAS_TRIAL_A", 50))
BATCH_SIZE_A    = 16
ALPHA_A         = 0.1          # const_ratio de RandomScheduling: la pérdida 'facl-0.1'
N_NIVELES       = 3            # profundidad fija: la búsqueda cubre solo filtros base, dropout y kernel por nivel
WARMUP_PRUNER_A = int(float(os.environ.get("WARMUP_FRAC", 0.6)) * EPOCAS_TRIAL_A)       # el pruner recién actúa pasado el 60% del cronograma

# ==========================================
# Igual que el código oficial (train_meteo.py)
# ==========================================
LR_OFICIAL      = float(os.environ.get("LR", 1e-3))   # default de --lr = 1e-3. LR solo se cambia en las pruebas de diagnóstico:
                                                      # optuna_A_oficial.sbatch hace `unset LR`, así que la corrida real usa 1e-3
ETA_MIN_OFICIAL = 1e-6         # CosineAnnealingLR(..., eta_min=1e-6)
MICRO_BATCH     = 1            # default de --micro_batch: sin acumulación de gradiente, un sorteo FAL/FCL por lote
CHEQUEO_NAN_CADA = 50          # cada cuántos pasos se mira si la pérdida ya es NaN/Inf (ver entrenar_una_epoca)

# MAE de utilspp.py, línea 302 (es la métrica 'mae' de MetricListEvaluator)
mae = lambda *args: torch.nn.functional.l1_loss(*args).cpu().detach().numpy()

# Errores de memoria: según la versión de PyTorch existe torch.OutOfMemoryError y/o torch.xpu.OutOfMemoryError
_OOM_TIPOS = tuple(t for t in (getattr(torch, "OutOfMemoryError", None),
                               getattr(getattr(torch, "xpu", None), "OutOfMemoryError", None))
                   if isinstance(t, type) and issubclass(t, BaseException))


def _es_oom(e):
    if _OOM_TIPOS and isinstance(e, _OOM_TIPOS):
        return True
    msg = str(e).lower()
    return "out of memory" in msg or "out_of_device_memory" in msg or "out_of_host_memory" in msg


def _vaciar_cache():
    try:
        if device.type == "xpu":
            torch.xpu.empty_cache()
    except Exception:
        pass


def _chequear_formas(y_pred, y):
    # Sin este chequeo, formas distintas (p. ej. (B,1,H,W) contra (B,H,W)) se broadcastearían en silencio
    if y_pred.shape != y.shape:
        raise ValueError(f"forma de la predicción {tuple(y_pred.shape)} != forma del objetivo {tuple(y.shape)}")


class PerdidaNoFinita(Exception):
    """La pérdida de entrenamiento se volvió NaN/Inf: los pesos ya quedaron arruinados y el trial no se recupera."""
    def __init__(self, paso):
        super().__init__(f"pérdida no finita detectada en el paso {paso}")
        self.paso = paso


def entrenar_una_epoca(modelo, loader, criterion, optimizer, scheduler, total_step):
    """Una época con el bucle de train_meteo.py (líneas 127-168). Devuelve (total_step, pérdida media de la época).

    Lo único que se agrega al bucle oficial es un chequeo cada CHEQUEO_NAN_CADA pasos: si la pérdida acumulada ya es
    NaN/Inf se corta el trial (PerdidaNoFinita) en lugar de seguir entrenando hasta el final de la época. No cambia
    el entrenamiento: con un NaN en la pérdida, el código oficial deja los pesos en NaN y el trial ya no se recupera.
    """
    suma, n = None, 0
    for x, y in loader:
        optimizer.zero_grad()
        modelo.train()
        x, y = x.to(device), y.to(device)
        y_pred = modelo(x)
        _chequear_formas(y_pred, y)
        # (B, C, H, W) -> (B, T=1, C, H, W): la forma que espera RandomScheduling (`_, _, _, H, W = pred.shape`)
        loss = criterion(y_pred.unsqueeze(1), y.unsqueeze(1))
        loss = loss / MICRO_BATCH
        loss.backward()
        total_step += 1
        if total_step % MICRO_BATCH == 0:
            optimizer.step()
            scheduler.step()
        suma = loss.detach() if suma is None else suma + loss.detach()     # se acumula en el dispositivo: sin sincronizar en cada paso
        n += 1
        if n % CHEQUEO_NAN_CADA == 0 and not torch.isfinite(suma).item():
            raise PerdidaNoFinita(total_step)
    perdida_media = float((suma / n).item())
    if not np.isfinite(perdida_media):
        raise PerdidaNoFinita(total_step)
    return total_step, perdida_media


def validar_mae(modelo, loader):
    """MAE de validación como MetricListEvaluator(['mae', ...]): suma de F.l1_loss por lote / cantidad de lotes."""
    modelo.eval()
    acumulado, batch_count = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_pred = modelo(x)
            _chequear_formas(y_pred, y)
            batch_count += 1
            acumulado += mae(y_pred, y)
    return float(acumulado / batch_count)


def objective_arquitectura(trial):
    tiempo_inicio_trial = time.time()

    # ==========================================
    # HIPERPARÁMETROS DE ARQUITECTURA (igual que el primer estudio)
    # ==========================================
    n_niveles      = N_NIVELES
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
    # DATALOADERS (igual que el primer estudio)
    # ==========================================
    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader_trial = DataLoader(train_dataset, batch_size=BATCH_SIZE_A,
                                    shuffle=True, generator=generator,
                                    num_workers=0, pin_memory=False)
    val_loader_trial   = DataLoader(val_dataset, batch_size=BATCH_SIZE_A,
                                    shuffle=False, num_workers=0, pin_memory=False)

    # ==========================================
    # MODELO, LOSS, OPTIMIZER Y SCHEDULER (código oficial)
    # ==========================================
    torch.manual_seed(SEED + trial.number)
    np.random.seed(SEED + trial.number)
    random.seed(SEED + trial.number)             # el sorteo FAL/FCL de RandomScheduling usa random.random()
    modelo_trial = UNet_WRF(
        in_channels=N_CANALES,                   # 9 (dataset temporal)
        n_niveles=n_niveles, n_filtros_base=n_filtros_base,
        kernel_sizes=kernel_sizes, dropout_ps=dropout_ps
    ).to(device)
    n_parametros = sum(p.numel() for p in modelo_trial.parameters())

    pasos_totales   = len(train_loader_trial) * EPOCAS_TRIAL_A / MICRO_BATCH                       # args.step, train_meteo.py:82
    criterion_trial = RandomScheduling(pasos_totales, MICRO_BATCH, const_ratio=ALPHA_A)            # train_meteo.py:37
    optimizer_trial = torch.optim.AdamW(modelo_trial.parameters(), lr=LR_OFICIAL)                  # train_meteo.py:99
    scheduler_trial = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_trial, T_max=pasos_totales,
                                                                 eta_min=ETA_MIN_OFICIAL)          # train_meteo.py:116

    mejor_objetivo = float('inf')
    mejor_epoca    = 0
    total_step     = 0

    # ==========================================
    # BUCLE DE ENTRENAMIENTO DEL TRIAL
    # ==========================================
    try:
        for epoch in range(EPOCAS_TRIAL_A):
            total_step, train_loss = entrenar_una_epoca(modelo_trial, train_loader_trial, criterion_trial,
                                                        optimizer_trial, scheduler_trial, total_step)
            epoch_val_mae = validar_mae(modelo_trial, val_loader_trial)

            # Detectar NaN/Inf y podar el trial en lugar de dejarlo completarse con basura
            if not np.isfinite(epoch_val_mae):
                trial.set_user_attr('nan_val_en_epoca', epoch + 1)
                print(f"[Trial {trial.number}] val_MAE no finito ({epoch_val_mae}) — podando trial.", flush=True)
                raise optuna.TrialPruned()

            if epoch_val_mae < mejor_objetivo:
                mejor_objetivo = epoch_val_mae
                mejor_epoca    = epoch + 1

            i_prob = min(criterion_trial.step, len(criterion_trial.prob_thres) - 1)
            print(f"[Trial {trial.number}] Epoch {epoch+1}/{EPOCAS_TRIAL_A} | val_MAE={epoch_val_mae:.5f} "
                  f"| train_loss={train_loss:.4f} | lr={optimizer_trial.param_groups[0]['lr']:.3e} "
                  f"| P(FCL)={float(criterion_trial.prob_thres[i_prob]):.3f} | paso {total_step}/{pasos_totales:.0f}", flush=True)

            trial.report(epoch_val_mae, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr('tiempo_trial_seg', time.time() - tiempo_inicio_trial)
        trial.set_user_attr('n_parametros', n_parametros)
        trial.set_user_attr('mejor_epoca', mejor_epoca)
        trial.set_user_attr('val_final', epoch_val_mae)
        trial.set_user_attr('criterio_val', 'MAE (normalizado)')
        trial.set_user_attr('pasos_totales', total_step)
        trial.set_user_attr('lr_final', optimizer_trial.param_groups[0]['lr'])
        trial.set_user_attr('n_canales', N_CANALES)
        trial.set_user_attr('dataset', DATASET_DIR)
        return mejor_objetivo

    except optuna.TrialPruned:
        raise

    except PerdidaNoFinita as e:
        trial.set_user_attr('nan_en_paso', e.paso)
        trial.set_user_attr('nan_en_epoca', e.paso / len(train_loader_trial))
        print(f"[Trial {trial.number}] pérdida no finita en el paso {e.paso} "
              f"(época {e.paso / len(train_loader_trial):.2f}) — podando trial.", flush=True)
        raise optuna.TrialPruned()

    except Exception as e:
        _vaciar_cache()
        if _es_oom(e):
            print(f"[Trial {trial.number}] OOM ({type(e).__name__}) — se descarta este trial.", flush=True)
            raise optuna.TrialPruned()
        # Cualquier otro error queda como FAIL (visible en el estudio) en vez de esconderse como podado
        print(f"[Trial {trial.number}] Error inesperado: {type(e).__name__}: {e} — el trial queda en FAIL.", flush=True)
        raise
