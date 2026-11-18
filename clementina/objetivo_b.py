"""Etapa B: optimización de hiperparámetros de ENTRENAMIENTO (arquitectura fija).

Por trial se guarda, en etapa_B/trial_XXXX/:
  best.pt    pesos de la época con mejor val_loss
  last.pt    pesos de la última época (se escribe al terminar el trial)
  resume.pt  estado completo (se reescribe cada época) -> permite reanudar si Slurm corta el job
  curva.csv  train_loss, val_loss, val_fal, val_fcl, P(t) y lr por época

Criterio de optimización: igual que en la Etapa A (mínimo del val_loss sobre las épocas).
Como se guardan best.pt y last.pt de cada trial, se puede re-rankear después con otra
métrica sin reentrenar.
"""
import os, time, csv, math
import numpy as np
import torch
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader

from modelo import UNet_WRF, FACL
from datos import train_dataset, val_dataset

SEED = 42
device = torch.device(os.environ.get("DEVICE", "xpu"))

# ==========================================
# ARQUITECTURA FIJA (ganadora de la Etapa A)
# ==========================================
ARQ_FINAL = dict(n_niveles=3, n_filtros_base=64, kernel_sizes=[5, 5, 3],
                 dropout_ps=[0.03, 0.08, 0.26])

# ==========================================
# ESPACIO DE BÚSQUEDA
# ==========================================
SMOKE = os.environ.get("SMOKE", "0") == "1"
LR_RANGO        = (1e-5, 1e-3)                      # log
ALPHA_RANGO     = (0.0, 0.30)
EPOCAS_OPCIONES = [2, 3] if SMOKE else list(range(20, 201, 20))   # 20, 40, ..., 200
BATCH_OPCIONES  = [8, 16, 32]
BATCH_VAL_B     = 16        # la validación acumula sobre todo el set: no depende de este valor
LR_MIN_SCHED    = 1e-7

DIR_B = os.environ.get("DIR_B", os.path.expanduser("~/lightning/etapa_B"))


def _guardar_atomico(obj, ruta):
    """Escribe a un temporal y renombra, para no dejar archivos a medias si Slurm mata el proceso."""
    tmp = ruta + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, ruta)


def _pesos_cpu(modelo):
    return {k: v.detach().cpu() for k, v in modelo.state_dict().items()}


def _escribir_curva(ruta, historial):
    if not historial:
        return
    tmp = ruta + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(historial[0].keys()))
        w.writeheader()
        w.writerows(historial)
    os.replace(tmp, ruta)


def _mismos_hparams(h, lr, alpha, epocas, batch_size):
    return (math.isclose(h["lr"], lr, rel_tol=1e-9) and math.isclose(h["alpha"], alpha, rel_tol=1e-9)
            and h["epocas"] == epocas and h["batch_size"] == batch_size and h["arq"] == ARQ_FINAL)


def objective_entrenamiento(trial):
    t_inicio = time.time()

    # ==========================================
    # HIPERPARÁMETROS DE ENTRENAMIENTO A OPTIMIZAR
    # ==========================================
    lr         = trial.suggest_float('lr', *LR_RANGO, log=True)
    alpha      = trial.suggest_float('alpha', *ALPHA_RANGO)
    epocas     = trial.suggest_categorical('epocas', EPOCAS_OPCIONES)
    batch_size = trial.suggest_categorical('batch_size', BATCH_OPCIONES)

    # Carpeta del trial. Un trial re-encolado tras un corte sigue usando la de su antecesor.
    dir_id  = int(trial.user_attrs.get("resume_from", trial.number))
    carpeta = os.path.join(DIR_B, f"trial_{dir_id:04d}")
    os.makedirs(carpeta, exist_ok=True)
    r_resume = os.path.join(carpeta, "resume.pt")
    r_best   = os.path.join(carpeta, "best.pt")
    r_last   = os.path.join(carpeta, "last.pt")
    r_curva  = os.path.join(carpeta, "curva.csv")
    trial.set_user_attr("dir_id", dir_id)
    trial.set_user_attr("carpeta", carpeta)

    # ==========================================
    # DATALOADERS, MODELO, LOSS, OPTIMIZER, SCHEDULER
    # ==========================================
    generator = torch.Generator()
    generator.manual_seed(SEED + dir_id)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              generator=generator, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_VAL_B, shuffle=False,
                              num_workers=0, pin_memory=False)

    torch.manual_seed(SEED + dir_id)
    np.random.seed(SEED + dir_id)               # sorteo FAL/FCL reproducible por trial
    modelo = UNet_WRF(in_channels=3, **ARQ_FINAL).to(device)
    n_parametros = sum(p.numel() for p in modelo.parameters())

    criterion = FACL(total_steps=epocas * len(train_loader), alpha=alpha).to(device)
    optimizer = optim.Adam(modelo.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epocas, eta_min=LR_MIN_SCHED)

    epoca_inicio, mejor_val, mejor_epoca = 0, float('inf'), 0
    val_fal_mejor = val_fcl_mejor = None
    historial, t_previo = [], 0.0

    # ==========================================
    # REANUDACIÓN (si hay un resume.pt compatible)
    # ==========================================
    if os.path.exists(r_resume):
        ck = torch.load(r_resume, map_location='cpu', weights_only=False)
        if _mismos_hparams(ck['hparams'], lr, alpha, epocas, batch_size):
            modelo.load_state_dict(ck['modelo'])
            optimizer.load_state_dict(ck['optimizer'])
            scheduler.load_state_dict(ck['scheduler'])
            criterion.step = ck['criterion_step']
            np.random.set_state(ck['np_rng'])
            torch.set_rng_state(ck['torch_rng'])
            generator.set_state(ck['gen_state'])
            epoca_inicio, mejor_val, mejor_epoca = ck['epoca'], ck['mejor_val'], ck['mejor_epoca']
            val_fal_mejor, val_fcl_mejor = ck['val_fal_mejor'], ck['val_fcl_mejor']
            historial, t_previo = ck['historial'], ck['tiempo_acum']
            trial.set_user_attr("reanudado_desde_epoca", epoca_inicio)
            print(f"[Trial {trial.number}] REANUDADO desde la época {epoca_inicio}/{epocas} ({carpeta})", flush=True)
        else:
            print(f"[Trial {trial.number}] resume.pt con hiperparámetros distintos: se ignora y se empieza de cero.", flush=True)

    # ==========================================
    # BUCLE DE ENTRENAMIENTO
    # ==========================================
    try:
        for epoch in range(epoca_inicio, epocas):
            t_ep = time.time()
            lr_ep = optimizer.param_groups[0]['lr']
            modelo.train()
            suma = torch.zeros((), device=device)
            n_b = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(modelo(batch_X), batch_y, mode='train')
                loss.backward()
                optimizer.step()
                suma += loss.detach()
                n_b += 1
            train_loss = (suma / max(n_b, 1)).item()

            # Validación sobre TODO el set (FCL global: se acumulan las sumas)
            modelo.eval()
            criterion.val_reset()
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    criterion.val_acumular(modelo(batch_X), batch_y)
            val = criterion.val_resultado()
            val_fal, val_fcl = criterion.val_fal, criterion.val_fcl

            if not np.isfinite(val):
                print(f"[Trial {trial.number}] val_loss no finito ({val}) — podando trial.", flush=True)
                raise optuna.TrialPruned()

            scheduler.step()

            if val < mejor_val:
                mejor_val, mejor_epoca = val, epoch + 1
                val_fal_mejor, val_fcl_mejor = val_fal, val_fcl
                _guardar_atomico({'modelo': _pesos_cpu(modelo), 'epoca': epoch + 1, 'val': val}, r_best)

            historial.append(dict(epoca=epoch + 1, train_loss=train_loss, val_loss=val, val_fal=val_fal,
                                  val_fcl=val_fcl, p_fcl=criterion.prob_fcl(), lr=lr_ep,
                                  seg_epoca=time.time() - t_ep))
            trial.report(val, epoch)

            _guardar_atomico({
                'hparams': dict(lr=lr, alpha=alpha, epocas=epocas, batch_size=batch_size, arq=ARQ_FINAL),
                'modelo': modelo.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'criterion_step': criterion.step,
                'np_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                'gen_state': generator.get_state(), 'epoca': epoch + 1,
                'mejor_val': mejor_val, 'mejor_epoca': mejor_epoca,
                'val_fal_mejor': val_fal_mejor, 'val_fcl_mejor': val_fcl_mejor,
                'historial': historial, 'tiempo_acum': t_previo + (time.time() - t_inicio),
            }, r_resume)
            _escribir_curva(r_curva, historial)

            print(f"[Trial {trial.number}] Epoch {epoch+1}/{epocas} | train={train_loss:.4f} | val={val:.4f} "
                  f"(FAL={val_fal:.5f} FCL={val_fcl:.4f}) | P(FCL)={criterion.prob_fcl():.2f} | "
                  f"{time.time()-t_ep:.0f}s", flush=True)

        # Fin normal (también cubre el caso de un resume.pt que ya estaba en la última época)
        ult = historial[-1]
        _guardar_atomico({'modelo': _pesos_cpu(modelo), 'epoca': epocas, 'val': ult['val_loss']}, r_last)
        _escribir_curva(r_curva, historial)

        trial.set_user_attr('tiempo_trial_seg', t_previo + (time.time() - t_inicio))
        trial.set_user_attr('n_parametros', n_parametros)
        trial.set_user_attr('mejor_epoca', mejor_epoca)
        trial.set_user_attr('val_final', ult['val_loss'])
        trial.set_user_attr('val_fal_mejor', val_fal_mejor)
        trial.set_user_attr('val_fcl_mejor', val_fcl_mejor)
        trial.set_user_attr('val_fal_final', ult['val_fal'])
        trial.set_user_attr('val_fcl_final', ult['val_fcl'])
        return mejor_val

    except optuna.TrialPruned:
        raise
    except Exception as e:
        # Se deja propagar (el trial queda FAIL y el worker sigue): no se disfraza de "podado".
        print(f"[Trial {trial.number}] ERROR {type(e).__name__}: {e}", flush=True)
        if device.type == "xpu":
            torch.xpu.empty_cache()
        raise
