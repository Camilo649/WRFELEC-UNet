"""Etapa B con la FACL OFICIAL: optimización de hiperparámetros de ENTRENAMIENTO con la arquitectura fija elegida en
la Etapa A oficial (trial #118: 3 niveles, 32 filtros, kernels [5, 5, 5]) y el dataset temporal de 9 canales.

Es objetivo_B_temporal.py con la pérdida y el entrenamiento del código oficial (https://github.com/argenycw/FACL,
commit f46009f, train_meteo.py), igual que objetivo_A_oficial.py en la Etapa A:
  - Pérdida: RandomScheduling oficial copiada textual (facl_oficial.py), un sorteo FAL/FCL por lote
    (micro_batch = 1), FCL con una sola raíz y sin eps, pérdida x sqrt(H*W). Su const_ratio es el `alpha`
    que se busca (en la Etapa A oficial era fijo, 0.1).
  - Optimizador: AdamW con los argumentos por defecto salvo lr, que se busca.
  - Tasa de aprendizaje: CosineAnnealingLR(T_max = pasos totales, eta_min = 1e-6), un scheduler.step() por cada
    optimizer.step() (train_meteo.py:116 y 164-168), igual que la Etapa A oficial.
  - Criterio de Optuna y "mejor modelo": menor MAE de validación (F.l1_loss promediado por lote, sobre los valores
    normalizados), con lotes de validación de 16 como en la Etapa A oficial (BATCH_VAL_B).
Lo que NO cambia respecto de la Etapa B temporal: espacio de búsqueda (lr, alpha, épocas, batch_size), dataset de 9
canales, el CSI y el FSS por época en curva.csv (metricas_B.py), el corte por NaN, OOM -> podado y la reanudación desde resume.pt.

Por trial, en etapa_B_oficial/trial_XXXX/:
  best.pt              pesos de la época con menor MAE de validación (+ su MAE, CSI y FSS)
  best_csi.pt          pesos de la época con mayor CSI medio de validación (csi_media)
  best_fss.pt          pesos de la época con mayor FSS de referencia (fss_u15_v3: umbral 15, ventana 3x3)
  last.pt              pesos de la última época (se escribe al terminar el trial)
  resume.pt            estado completo (se reescribe cada época) -> permite reanudar si Slurm corta el job
  curva.csv            train_loss, val_loss (MAE), P(FCL), lr, CSI y FSS por época
"""
import os, time, csv, math, random
import numpy as np
import torch
import torch.nn.functional as F
import optuna
from torch.utils.data import DataLoader

from modelo import UNet_WRF                    # la misma U-Net (no se usa su FACL)
from facl_oficial import RandomScheduling      # copia textual de utilspp.RandomScheduling
from datos import train_dataset, val_dataset, DATASET_DIR
from metricas_B import AcumuladorMetricas, columnas_metricas

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
# ARQUITECTURA FIJA (trial #118 de la Etapa A oficial: 5 410 241 parámetros)
# Dropout: los del #118 (0.0828 / 0.0635 / 0.2037) redondeados a 2 decimales.
# ==========================================
ARQ_FINAL = dict(n_niveles=3, n_filtros_base=32, kernel_sizes=[5, 5, 5],
                 dropout_ps=[0.08, 0.06, 0.20])
N_PARAMETROS_ARQ = 5410241

# ==========================================
# ESPACIO DE BÚSQUEDA (igual que la Etapa B temporal)
# ==========================================
SMOKE = os.environ.get("SMOKE", "0") == "1"
LR_RANGO        = (1e-5, 1e-3)                      # log; el lr oficial (1e-3) es el extremo superior
ALPHA_RANGO     = (0.0, 0.30)                       # const_ratio de RandomScheduling
EPOCAS_OPCIONES = [2, 3] if SMOKE else list(range(20, 201, 20))   # 20, 40, ..., 200
BATCH_OPCIONES  = [8, 16, 32]
BATCH_VAL_B     = 16        # el MAE oficial se promedia por lote: se fija en 16 (como la Etapa A oficial)
ETA_MIN_OFICIAL = 1e-6      # CosineAnnealingLR(..., eta_min=1e-6)
MICRO_BATCH     = 1         # default de --micro_batch: sin acumulación de gradiente, un sorteo FAL/FCL por lote
CHEQUEO_NAN_CADA = 50       # cada cuántos lotes se mira si la pérdida de entrenamiento ya es NaN/Inf
FSS_REF = "fss_u15_v3"      # FSS de referencia: umbral 15, ventana 3x3

DIR_B = os.environ.get("DIR_B", os.path.expanduser("~/lightning/etapa_B_oficial"))

print(f"[objetivo_B_oficial] Dataset: {DATASET_DIR} | canales: {N_CANALES} | device: {device} | "
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


def validar(modelo, val_loader):
    """Una pasada sobre todo el set de validación: MAE como MetricListEvaluator (suma de F.l1_loss por lote /
    cantidad de lotes, utilspp.py:302 y eval.py:158-178) + CSI y FSS."""
    modelo.eval()
    acum = AcumuladorMetricas(desnormalizar_y, device)
    suma_mae, n_lotes = 0.0, 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            pred = modelo(batch_X)
            if pred.shape != batch_y.shape:
                raise ValueError(f"forma de la predicción {tuple(pred.shape)} != forma del objetivo {tuple(batch_y.shape)}")
            suma_mae += float(F.l1_loss(pred, batch_y).cpu().detach().numpy())
            n_lotes += 1
            acum.acumular(pred, batch_y)
    return suma_mae / n_lotes, acum.resultados()


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
    r_best_csi = os.path.join(carpeta, "best_csi.pt")
    r_best_fss = os.path.join(carpeta, "best_fss.pt")
    r_last   = os.path.join(carpeta, "last.pt")
    r_curva  = os.path.join(carpeta, "curva.csv")
    trial.set_user_attr("dir_id", dir_id)
    trial.set_user_attr("carpeta", carpeta)

    # ==========================================
    # DATALOADERS, MODELO, LOSS, OPTIMIZER, SCHEDULER (código oficial)
    # ==========================================
    generator = torch.Generator()
    generator.manual_seed(SEED + dir_id)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              generator=generator, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_VAL_B, shuffle=False,
                              num_workers=0, pin_memory=False)

    torch.manual_seed(SEED + dir_id)
    np.random.seed(SEED + dir_id)
    random.seed(SEED + dir_id)                  # el sorteo FAL/FCL de RandomScheduling usa random.random()
    modelo = UNet_WRF(in_channels=N_CANALES, **ARQ_FINAL).to(device)
    n_parametros = sum(p.numel() for p in modelo.parameters())

    pasos_totales = len(train_loader) * epocas / MICRO_BATCH                         # args.step, train_meteo.py:82
    criterion = RandomScheduling(pasos_totales, MICRO_BATCH, const_ratio=alpha)      # train_meteo.py:37
    optimizer = torch.optim.AdamW(modelo.parameters(), lr=lr)                        # train_meteo.py:99
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pasos_totales,
                                                           eta_min=ETA_MIN_OFICIAL)  # train_meteo.py:116

    epoca_inicio, mejor_val, mejor_epoca = 0, float('inf'), 0
    total_step = 0
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
            criterion.step, criterion.out = ck['criterion_step'], ck['criterion_out']
            total_step = ck['total_step']
            random.setstate(ck['py_rng'])
            np.random.set_state(ck['np_rng'])
            torch.set_rng_state(ck['torch_rng'])
            generator.set_state(ck['gen_state'])
            epoca_inicio, mejor_val, mejor_epoca = ck['epoca'], ck['mejor_val'], ck['mejor_epoca']
            historial, t_previo = ck['historial'], ck['tiempo_acum']
            trial.set_user_attr("reanudado_desde_epoca", epoca_inicio)
            print(f"[Trial {trial.number}] REANUDADO desde la época {epoca_inicio}/{epocas} ({carpeta})", flush=True)
        else:
            print(f"[Trial {trial.number}] resume.pt con hiperparámetros distintos: se ignora y se empieza de cero.", flush=True)

    # Mejor CSI y mejor FSS vistos hasta ahora (al reanudar se recalculan desde el historial; el primero que alcanza
    # el máximo gana, igual que max() sobre curva.csv)
    mejor_csi = max((h['csi_media'] for h in historial if np.isfinite(h['csi_media'])), default=-float('inf'))
    mejor_fss = max((h[FSS_REF] for h in historial if np.isfinite(h[FSS_REF])), default=-float('inf'))

    # ==========================================
    # BUCLE DE ENTRENAMIENTO (train_meteo.py:127-168)
    # ==========================================
    try:
        for epoch in range(epoca_inicio, epocas):
            t_ep = time.time()
            suma = torch.zeros((), device=device)
            n_b = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                modelo.train()
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                y_pred = modelo(batch_X)
                if y_pred.shape != batch_y.shape:
                    raise ValueError(f"forma de la predicción {tuple(y_pred.shape)} != forma del objetivo {tuple(batch_y.shape)}")
                # (B, C, H, W) -> (B, T=1, C, H, W): la forma que espera RandomScheduling (`_, _, _, H, W = pred.shape`)
                loss = criterion(y_pred.unsqueeze(1), batch_y.unsqueeze(1))
                loss = loss / MICRO_BATCH
                loss.backward()
                total_step += 1
                if total_step % MICRO_BATCH == 0:
                    optimizer.step()
                    scheduler.step()
                suma += loss.detach()
                n_b += 1
                if n_b % CHEQUEO_NAN_CADA == 0 and not torch.isfinite(suma).item():
                    raise PerdidaNoFinita(epoch + 1, n_b)
            train_loss = (suma / max(n_b, 1)).item()
            if not np.isfinite(train_loss):
                raise PerdidaNoFinita(epoch + 1, n_b)

            # Validación sobre TODO el set: MAE (criterio oficial) + CSI y FSS en la misma pasada
            val, metricas = validar(modelo, val_loader)

            if not np.isfinite(val):
                trial.set_user_attr('nan_val_en_epoca', epoch + 1)
                print(f"[Trial {trial.number}] val_MAE no finito ({val}) — podando trial.", flush=True)
                raise optuna.TrialPruned()

            lr_ep = optimizer.param_groups[0]['lr']      # lr al terminar la época (el scheduler avanza por paso)
            i_prob = min(criterion.step, len(criterion.prob_thres) - 1)
            p_fcl = float(criterion.prob_thres[i_prob])

            if val < mejor_val:
                mejor_val, mejor_epoca = val, epoch + 1
                _guardar_atomico({'modelo': _pesos_cpu(modelo), 'epoca': epoch + 1, 'val': val, **metricas}, r_best)

            # Pesos de la época con mejor CSI y de la época con mejor FSS (solo si mejoran estrictamente)
            for clave, ruta, mejor in (('csi_media', r_best_csi, mejor_csi), (FSS_REF, r_best_fss, mejor_fss)):
                if np.isfinite(metricas[clave]) and metricas[clave] > mejor:
                    _guardar_atomico({'modelo': _pesos_cpu(modelo), 'epoca': epoch + 1, 'val': val, **metricas}, ruta)
                    if clave == 'csi_media':
                        mejor_csi = metricas[clave]
                    else:
                        mejor_fss = metricas[clave]

            historial.append(dict(epoca=epoch + 1, train_loss=train_loss, val_loss=val, p_fcl=p_fcl, lr=lr_ep,
                                  seg_epoca=time.time() - t_ep, **metricas))
            trial.report(val, epoch)

            _guardar_atomico({
                'hparams': dict(lr=lr, alpha=alpha, epocas=epocas, batch_size=batch_size, arq=ARQ_FINAL),
                'modelo': modelo.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'criterion_step': criterion.step,
                'criterion_out': criterion.out, 'total_step': total_step,
                'py_rng': random.getstate(), 'np_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                'gen_state': generator.get_state(), 'epoca': epoch + 1,
                'mejor_val': mejor_val, 'mejor_epoca': mejor_epoca,
                'historial': historial, 'tiempo_acum': t_previo + (time.time() - t_inicio),
            }, r_resume)
            _escribir_curva(r_curva, historial)

            print(f"[Trial {trial.number}] Epoch {epoch+1}/{epocas} | train={train_loss:.4f} | val_MAE={val:.5f} "
                  f"| CSI={metricas['csi_media']:.4f} FSS(u15,v3)={metricas[FSS_REF]:.4f} | lr={lr_ep:.3e} "
                  f"| P(FCL)={p_fcl:.3f} | {time.time()-t_ep:.0f}s", flush=True)

        # Fin normal (también cubre el caso de un resume.pt que ya estaba en la última época)
        ult = historial[-1]
        _guardar_atomico({'modelo': _pesos_cpu(modelo), 'epoca': epocas, 'val': ult['val_loss'],
                          **{c: ult[c] for c in columnas_metricas()}}, r_last)
        _escribir_curva(r_curva, historial)

        csis = [(h['csi_media'], h['epoca']) for h in historial if np.isfinite(h['csi_media'])]
        fsss = [(h[FSS_REF], h['epoca']) for h in historial if np.isfinite(h[FSS_REF])]
        trial.set_user_attr('tiempo_trial_seg', t_previo + (time.time() - t_inicio))
        trial.set_user_attr('n_parametros', n_parametros)
        trial.set_user_attr('mejor_epoca', mejor_epoca)
        trial.set_user_attr('val_final', ult['val_loss'])
        trial.set_user_attr('criterio_val', 'MAE (normalizado)')
        trial.set_user_attr('pasos_totales', total_step)
        trial.set_user_attr('lr_final', ult['lr'])
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
