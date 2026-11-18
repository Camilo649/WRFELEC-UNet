"""Etapa B TEMPORAL: optimización de hiperparámetros de ENTRENAMIENTO con la arquitectura fija elegida en la
Etapa A temporal (trial #56) y el dataset de 9 canales.

Es objetivo_b.py con estos cambios (todo lo demás queda igual: FACL de modelo.py con validación 0.5·FAL + 0.5·FCL
sobre todo el set, Adam, cosine por época hasta 1e-7, espacio de búsqueda, reanudación desde resume.pt):
  1. Dataset temporal: in_channels sale de stats.npz y se exige que sean 9 (objetivo_b.py tenía in_channels=3).
  2. ARQ_FINAL = trial #56 de la Etapa A temporal: 3 niveles, 64 filtros, kernels [9, 5, 3] y dropout
     [0.08, 0.11, 0.30] (los del #56 redondeados a 2 decimales).
  3. Carpeta propia (etapa_B_temporal/) para no mezclarse con los trials de la Etapa B anterior.
  4. Se guardan los pesos de TODAS las épocas (epocas/epoca_XXX.pt, ~43 MB cada uno) además de best.pt, last.pt
     y resume.pt.
  5. En cada época se calculan sobre validación el CSI y la matriz FSS con las definiciones del notebook
     (metricas_B.py), en la misma pasada que el val_loss, y van al curva.csv y a cada epoca_XXX.pt.
     El criterio de Optuna sigue siendo el mínimo del val_loss. El FSS de referencia es el de umbral 15 y
     ventana 3x3 (fss_u15_v3): es el que se muestra en el log y en los user_attrs; las 16 celdas van al curva.csv.
  6. Si la pérdida de entrenamiento se vuelve NaN/Inf se poda el trial enseguida (se mira cada 50 lotes), y los
     errores de memoria se podan en vez de quedar FAIL (así el TPE aprende a evitar esa zona). Cualquier otro
     error sigue quedando FAIL, como antes.

Por trial, en etapa_B_temporal/trial_XXXX/:
  epocas/epoca_XXX.pt  pesos de cada época (+ su val_loss, CSI y FSS)
  best.pt              pesos de la época con mejor val_loss
  last.pt              pesos de la última época (se escribe al terminar el trial)
  resume.pt            estado completo (se reescribe cada época) -> permite reanudar si Slurm corta el job
  curva.csv            train_loss, val_loss, val_fal, val_fcl, P(t), lr, CSI y FSS por época
"""
import os, time, csv, math
import numpy as np
import torch
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader

from modelo import UNet_WRF, FACL
from datos import train_dataset, val_dataset, DATASET_DIR
from metricas_B import AcumuladorMetricas

SEED = 42
device = torch.device(os.environ.get("DEVICE", "xpu"))

# ==========================================
# Chequeo de canales: el dataset temporal tiene 9 (si DATASET_DIR apunta al original, hay 3)
# ==========================================
_stats = np.load(os.path.join(DATASET_DIR, "stats.npz"))
N_CANALES = int(_stats["min_X"].shape[0])
assert N_CANALES == 9, (f"stats.npz de {DATASET_DIR} tiene {N_CANALES} canales (se esperaban 9): "
                        f"¿DATASET_DIR apunta al dataset original?")

# Desnormalización del propio dataset (con SUBSET, val_dataset es un Subset y el dataset está en .dataset)
desnormalizar_y = getattr(val_dataset, "dataset", val_dataset).desnormalizar_y

# ==========================================
# ARQUITECTURA FIJA (trial #56 de la Etapa A temporal)
# ==========================================
ARQ_FINAL = dict(n_niveles=3, n_filtros_base=64, kernel_sizes=[9, 5, 3],
                 dropout_ps=[0.08, 0.11, 0.30])

# ==========================================
# ESPACIO DE BÚSQUEDA (igual que objetivo_b.py)
# ==========================================
SMOKE = os.environ.get("SMOKE", "0") == "1"
LR_RANGO        = (1e-5, 1e-3)                      # log
ALPHA_RANGO     = (0.0, 0.30)
EPOCAS_OPCIONES = [2, 3] if SMOKE else list(range(20, 201, 20))   # 20, 40, ..., 200
BATCH_OPCIONES  = [8, 16, 32]
BATCH_VAL_B     = 16        # la validación acumula sobre todo el set: no depende de este valor
LR_MIN_SCHED    = 1e-7
CHEQUEO_NAN_CADA = 50       # cada cuántos lotes se mira si la pérdida de entrenamiento ya es NaN/Inf
FSS_REF = "fss_u15_v3"      # FSS de referencia: umbral 15, ventana 3x3

DIR_B = os.environ.get("DIR_B", os.path.expanduser("~/lightning/etapa_B_temporal"))

print(f"[objetivo_B_temporal] Dataset: {DATASET_DIR} | canales: {N_CANALES} | device: {device} | "
      f"arquitectura: {ARQ_FINAL} | carpeta: {DIR_B}", flush=True)

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


class PerdidaNoFinita(Exception):
    """La pérdida de entrenamiento se volvió NaN/Inf: los pesos ya quedaron arruinados."""
    def __init__(self, epoca, lote):
        super().__init__(f"pérdida no finita en la época {epoca}, lote {lote}")
        self.epoca, self.lote = epoca, lote


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


def validar(modelo, criterion, val_loader):
    """Una pasada sobre todo el set de validación: val_loss de la FACL (sumas acumuladas) + CSI y FSS."""
    modelo.eval()
    criterion.val_reset()
    acum = AcumuladorMetricas(desnormalizar_y, device)
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            pred = modelo(batch_X)
            criterion.val_acumular(pred, batch_y)
            acum.acumular(pred, batch_y)          # también chequea que pred y batch_y tengan la misma forma
    val = criterion.val_resultado()
    return val, criterion.val_fal, criterion.val_fcl, acum.resultados()


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
    dir_epocas = os.path.join(carpeta, "epocas")
    os.makedirs(dir_epocas, exist_ok=True)
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
    modelo = UNet_WRF(in_channels=N_CANALES, **ARQ_FINAL).to(device)
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

    # Si el trial empieza de cero en una carpeta que ya tiene épocas guardadas (de otra corrida), se apartan
    # (sin borrarlas) para que no se mezclen con las nuevas
    if epoca_inicio == 0 and any(f.endswith(".pt") for f in os.listdir(dir_epocas)):
        apartadas = os.path.join(carpeta, f"epocas_descartadas_{time.strftime('%Y%m%d_%H%M%S')}")
        os.replace(dir_epocas, apartadas)
        os.makedirs(dir_epocas)
        print(f"[Trial {trial.number}] había épocas de otra corrida en {dir_epocas}: movidas a {apartadas}", flush=True)

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
                if n_b % CHEQUEO_NAN_CADA == 0 and not torch.isfinite(suma).item():
                    raise PerdidaNoFinita(epoch + 1, n_b)
            train_loss = (suma / max(n_b, 1)).item()
            if not np.isfinite(train_loss):
                raise PerdidaNoFinita(epoch + 1, n_b)

            # Validación sobre TODO el set (FCL global: se acumulan las sumas) + CSI y FSS en la misma pasada
            val, val_fal, val_fcl, metricas = validar(modelo, criterion, val_loader)

            if not np.isfinite(val):
                print(f"[Trial {trial.number}] val_loss no finito ({val}) — podando trial.", flush=True)
                raise optuna.TrialPruned()

            scheduler.step()

            # Pesos de ESTA época (antes del resume.pt: si el job muere entre los dos, al reanudar se rehace la época)
            _guardar_atomico({'modelo': _pesos_cpu(modelo), 'epoca': epoch + 1, 'val': val,
                              'val_fal': val_fal, 'val_fcl': val_fcl, **metricas},
                             os.path.join(dir_epocas, f"epoca_{epoch + 1:03d}.pt"))

            if val < mejor_val:
                mejor_val, mejor_epoca = val, epoch + 1
                val_fal_mejor, val_fcl_mejor = val_fal, val_fcl
                _guardar_atomico({'modelo': _pesos_cpu(modelo), 'epoca': epoch + 1, 'val': val}, r_best)

            historial.append(dict(epoca=epoch + 1, train_loss=train_loss, val_loss=val, val_fal=val_fal,
                                  val_fcl=val_fcl, p_fcl=criterion.prob_fcl(), lr=lr_ep,
                                  seg_epoca=time.time() - t_ep, **metricas))
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
                  f"(FAL={val_fal:.5f} FCL={val_fcl:.4f}) | CSI={metricas['csi_media']:.4f} "
                  f"FSS(u15,v3)={metricas[FSS_REF]:.4f} | P(FCL)={criterion.prob_fcl():.2f} | "
                  f"{time.time()-t_ep:.0f}s", flush=True)

        # Fin normal (también cubre el caso de un resume.pt que ya estaba en la última época)
        ult = historial[-1]
        _guardar_atomico({'modelo': _pesos_cpu(modelo), 'epoca': epocas, 'val': ult['val_loss']}, r_last)
        _escribir_curva(r_curva, historial)

        csis = [(h['csi_media'], h['epoca']) for h in historial if np.isfinite(h['csi_media'])]
        fsss = [(h[FSS_REF], h['epoca']) for h in historial if np.isfinite(h[FSS_REF])]
        trial.set_user_attr('tiempo_trial_seg', t_previo + (time.time() - t_inicio))
        trial.set_user_attr('n_parametros', n_parametros)
        trial.set_user_attr('mejor_epoca', mejor_epoca)
        trial.set_user_attr('val_final', ult['val_loss'])
        trial.set_user_attr('val_fal_mejor', val_fal_mejor)
        trial.set_user_attr('val_fcl_mejor', val_fcl_mejor)
        trial.set_user_attr('val_fal_final', ult['val_fal'])
        trial.set_user_attr('val_fcl_final', ult['val_fcl'])
        if csis:
            csi_max, ep_csi = max(csis)
            trial.set_user_attr('csi_media_mejor', csi_max)
            trial.set_user_attr('csi_media_mejor_epoca', ep_csi)
        if fsss:
            fss_max, ep_fss = max(fsss)
            trial.set_user_attr('fss_u15_v3_mejor', fss_max)
            trial.set_user_attr('fss_u15_v3_mejor_epoca', ep_fss)
        return mejor_val

    except optuna.TrialPruned:
        raise
    except PerdidaNoFinita as e:
        trial.set_user_attr('nan_en_epoca', e.epoca)
        trial.set_user_attr('nan_en_lote', e.lote)
        print(f"[Trial {trial.number}] {e} — podando trial.", flush=True)
        raise optuna.TrialPruned()
    except Exception as e:
        _vaciar_cache()
        if _es_oom(e):
            trial.set_user_attr('oom', True)
            print(f"[Trial {trial.number}] OOM ({type(e).__name__}) con batch_size={batch_size} — podando trial.", flush=True)
            raise optuna.TrialPruned()
        # Se deja propagar (el trial queda FAIL y el worker sigue): no se disfraza de "podado".
        print(f"[Trial {trial.number}] ERROR {type(e).__name__}: {e}", flush=True)
        raise
