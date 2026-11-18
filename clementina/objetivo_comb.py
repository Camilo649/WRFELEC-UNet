"""Optimización de la Ext U-Net Comb V2.1 en dos fases, con el protocolo de las optimizaciones anteriores de la Comb
(Tablas 5.3 y 5.4 de la monografía y bitácora) y los ajustes que justifican las propias corridas de la V2.1 (BITACORA).

  ETAPA=A (Fase 1, arquitectura): 3 niveles FIJOS (reunión del 05/10); filtros del primer nivel 16/32/64; kernel de cada
      nivel 3/5/7/9; dropout de cada nivel en [0, 0.3]. Fijo (ver FIJOS_A): lr 1e-4, lote 16, MAX_PROB 0.75, w+ 17.53,
      rampa de 30 épocas (completa), 30 épocas con early stopping (paciencia 6). Poda: MedianPruner (calentamiento en el worker).
  ETAPA=B (Fase 2, entrenamiento): arquitectura fija leída de estudios/arq_comb_A.json (la deja arq_desde_A.py a partir del
      journal de la Etapa A). Se busca lr, lote, MAX_PROB y max_norm. Fijos (ver FIJOS_B): 150 épocas con early stopping
      (paciencia 15), w+ 17.53 y rampa de 60 épocas. w+ y la rampa van FIJOS: con el WMSE en escala real, el valor de la
      pérdida depende directamente de w+, y en las dos Etapas B de la V2.1 w+ explicó el 92-95 % de la importancia y se
      fue al piso del rango (reunión del 13/08).

Criterio de Optuna y "mejor época": la pérdida de validación propia de la Comb, (WMSE + sqrt(HW)*FAL + sqrt(HW)*FCL)/3
sobre todo el set (como en el notebook y en la monografía para los modelos híbridos). Además se guardan val_wmse (sin
normalizar), val_fal y val_fcl (sin escalar), y el CSI y el FSS por época, para poder re-rankear los ensayos por métricas.
Optimizador Adam con cosine annealing por época hasta 1e-7 (igual que objetivo_b.py de la Ext Clip FACL) y recorte de
gradientes con max_norm (fijo en 65 en la Etapa A, como en la Etapa A de la Comb V2; se busca en la B).

Por trial, en DIR/trial_XXXX/: best.pt (menor val), last.pt (última época, con sus métricas), resume.pt (cada época) y curva.csv.
"""
import os, json, time, csv, math, random
import numpy as np
import torch
import optuna
from torch.utils.data import DataLoader

from modelo import UNet_WRF
from perdida_comb import CombV21
from datos import train_dataset, val_dataset, DATASET_DIR
from metricas_B import AcumuladorMetricas, columnas_metricas

SEED = 42
ETAPA = os.environ.get("ETAPA", "B").upper()
assert ETAPA in ("A", "B"), f"ETAPA tiene que ser A o B (es {ETAPA!r})"
device = torch.device(os.environ.get("DEVICE", "xpu"))
SMOKE = os.environ.get("SMOKE", "0") == "1"

_stats = np.load(os.path.join(DATASET_DIR, "stats.npz"))
N_CANALES = int(_stats["min_X"].shape[0])
assert N_CANALES == 3, (f"stats.npz de {DATASET_DIR} tiene {N_CANALES} canales (se esperaban 3: MDBZ, HIELO, CTT): "
                        f"¿DATASET_DIR apunta al dataset temporal?")
desnormalizar_y = getattr(val_dataset, "dataset", val_dataset).desnormalizar_y

# ---------------- configuración ----------------
BATCH_VAL = 16
ETA_MIN = 1e-7                     # igual que objetivo_b.py (Ext Clip FACL): cosine POR ÉPOCA hasta 1e-7
CHEQUEO_NAN_CADA = 50
FSS_REF = "fss_u15_v3"
MIN_PROB = 0.0

# Fase 1 de la Comb V2 (Tabla 5.3): lr 1e-4, lote 16, MAX_PROB 0.75, 30 épocas, early stopping con paciencia 6,
# max_norm 65 (celda de la Etapa A del notebook). Ajustes: w+ = 17.53 (el de la Fase 2 de la Comb V2, fijo) y la rampa
# fija en vez de PROB_SLOPE = 1e-5 (reunión del 13/08): 30 épocas en la A (rampa completa) y 60 en la B.
W_POS_FIJO = 17.53
RAMPA_EPOCAS_FIJA = 60               # Etapa B (150 épocas)
EPOCAS_A = 3 if SMOKE else 30
FIJOS_A = dict(epocas=EPOCAS_A, batch_size=16, lr=1e-4, max_prob=0.75, w_pos=W_POS_FIJO,
               rampa_epocas=EPOCAS_A, max_norm=65.0, paciencia=6)   # rampa completa: Q(t) llega a MAX_PROB en la última época
NIVELES_A = [3]                      # niveles fijos (reunión del 05/10); el kernel/dropout se buscan por nivel
FILTROS_A = [16, 32, 64]             # 128 no entra en 18-46 h con el conjunto extendido (en la FACL costó >24 h por ensayo)
KERNELS_A = [3, 5, 7, 9]
DROPOUT_A = (0.0, 0.30)
if SMOKE:        # solo para la prueba rápida: espacio mínimo
    NIVELES_A, FILTROS_A, KERNELS_A = [3], [16], [3]

# Etapa B (Tabla 5.4 y Etapas B de la V2.1): se busca lr, lote, MAX_PROB y max_norm; w+ y la rampa van fijos
FIJOS_B = dict(epocas=3 if SMOKE else 150, paciencia=15, w_pos=W_POS_FIJO, rampa_epocas=RAMPA_EPOCAS_FIJA)
LR_RANGO = (1e-5, 1e-3)
BATCH_B = [8, 16, 32]                # la Tabla 5.4 incluía 4; con el conjunto extendido y 150 épocas es inviable
MAX_PROB_RANGO = (0.5, 0.95)
MAX_NORM_RANGO = (50.0, 10000.0)     # log
RAMPA_EXACTA = os.environ.get("RAMPA_EXACTA", "1") == "1"   # 1 (default): Q(t) llega a MAX_PROB justo en la época fijada (fórmula de la monografía); 0: fórmula del código del notebook (llega en E/MAX_PROB épocas)

DIR = os.environ.get("DIR_COMB", os.path.expanduser(f"~/lightning/etapa_{ETAPA}_comb"))
ARQ_JSON = os.environ.get("ARQ_JSON", os.path.expanduser("~/lightning/estudios/arq_comb_A.json"))

if ETAPA == "B":
    with open(ARQ_JSON) as f:
        ARQ_B = json.load(f)["arq"]
else:
    ARQ_B = None
print(f"[objetivo_comb] ETAPA {ETAPA} | dataset {DATASET_DIR} ({N_CANALES} canales) | device {device} | "
      f"rampa_exacta {RAMPA_EXACTA} | carpeta {DIR} | arq B: {ARQ_B}", flush=True)

_OOM_TIPOS = tuple(t for t in (getattr(torch, "OutOfMemoryError", None),
                               getattr(getattr(torch, "xpu", None), "OutOfMemoryError", None))
                   if isinstance(t, type) and issubclass(t, BaseException))


def _es_oom(e):
    if _OOM_TIPOS and isinstance(e, _OOM_TIPOS):
        return True
    m = str(e).lower()
    return "out of memory" in m or "out_of_device_memory" in m or "out_of_host_memory" in m


def _vaciar_cache():
    try:
        if device.type == "xpu":
            torch.xpu.empty_cache()
    except Exception:
        pass


class PerdidaNoFinita(Exception):
    def __init__(self, epoca, lote):
        super().__init__(f"pérdida no finita en la época {epoca}, lote {lote}")
        self.epoca, self.lote = epoca, lote


def _guardar_atomico(obj, ruta):
    tmp = ruta + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, ruta)


def _pesos_cpu(m):
    return {k: v.detach().cpu() for k, v in m.state_dict().items()}


def _escribir_curva(ruta, hist):
    if not hist:
        return
    tmp = ruta + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        w.writeheader()
        w.writerows(hist)
    os.replace(tmp, ruta)


def _mismos(h, hp):
    for k, v in hp.items():
        if isinstance(v, float):
            if not math.isclose(h[k], v, rel_tol=1e-9):
                return False
        elif h[k] != v:
            return False
    return True


def validar(modelo, criterion, val_loader):
    modelo.eval()
    acum = AcumuladorMetricas(desnormalizar_y, device)
    criterion.val_reset()
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            pred = modelo(X)
            if pred.shape != y.shape:
                raise ValueError(f"forma de la predicción {tuple(pred.shape)} != objetivo {tuple(y.shape)}")
            criterion.val_acumular(pred, y)
            acum.acumular(pred, y)
    lval = criterion.val_resultado()
    return lval, criterion.val_fal, criterion.val_fcl, criterion.val_wmse, acum.resultados()


def entrenar(trial, carpeta, arq, hp):
    """hp: lr, epocas, batch_size, max_prob, w_pos, rampa_epocas (en épocas), max_norm. Devuelve el mejor val."""
    t_inicio = time.time()
    epocas, batch_size = hp["epocas"], hp["batch_size"]
    paciencia = int(hp.get("paciencia", 0))     # 0 = sin early stopping
    os.makedirs(carpeta, exist_ok=True)
    r_resume, r_best, r_last, r_curva = (os.path.join(carpeta, n) for n in ("resume.pt", "best.pt", "last.pt", "curva.csv"))
    dir_id = int(trial.user_attrs.get("resume_from", trial.number))

    gen = torch.Generator()
    gen.manual_seed(SEED + dir_id)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=gen, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_VAL, shuffle=False, num_workers=0)
    torch.manual_seed(SEED + dir_id); np.random.seed(SEED + dir_id); random.seed(SEED + dir_id)

    modelo = UNet_WRF(in_channels=N_CANALES, **arq).to(device)
    n_par = sum(p.numel() for p in modelo.parameters())
    pasos_ep = len(train_loader)
    pasos_totales = pasos_ep * epocas
    criterion = CombV21(desnormalizar_y, peso_positivo=hp["w_pos"], max_prob=hp["max_prob"],
                        epocas_para_max_prob=hp["rampa_epocas"], pasos_por_epoca=pasos_ep, min_prob=MIN_PROB,
                        rampa_exacta=RAMPA_EXACTA)
    optimizer = torch.optim.Adam(modelo.parameters(), lr=hp["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epocas, eta_min=ETA_MIN)
    trial.set_user_attr("n_parametros", n_par)
    trial.set_user_attr("prob_slope", criterion.prob_slope)
    trial.set_user_attr("dir_id", dir_id)
    trial.set_user_attr("carpeta", carpeta)

    ep0, mejor_val, mejor_ep, hist, t_prev = 0, float("inf"), 0, [], 0.0
    hp_id = dict(hp, arq=arq)
    if os.path.exists(r_resume):
        ck = torch.load(r_resume, map_location="cpu", weights_only=False)
        if _mismos(ck["hparams"], hp_id):
            modelo.load_state_dict(ck["modelo"]); optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"]); criterion.step = ck["criterion_step"]
            random.setstate(ck["py_rng"]); np.random.set_state(ck["np_rng"]); torch.set_rng_state(ck["torch_rng"])
            gen.set_state(ck["gen_state"])
            ep0, mejor_val, mejor_ep, hist, t_prev = ck["epoca"], ck["mejor_val"], ck["mejor_epoca"], ck["historial"], ck["tiempo_acum"]
            trial.set_user_attr("reanudado_desde_epoca", ep0)
            print(f"[Trial {trial.number}] REANUDADO desde la época {ep0}/{epocas} ({carpeta})", flush=True)
        else:
            print(f"[Trial {trial.number}] resume.pt con hiperparámetros distintos: se ignora.", flush=True)

    try:
        if paciencia and ep0 > 0 and len(hist) - mejor_ep >= paciencia:
            ep0 = epocas          # el trial ya había terminado por early stopping antes del corte
        for epoch in range(ep0, epocas):
            t_ep = time.time()
            lr_ep = optimizer.param_groups[0]["lr"]     # lr de la época (como objetivo_b.py)
            suma = torch.zeros((), device=device)
            n_b = 0
            conteo = {"WMSE": 0, "FAL": 0, "FCL": 0}
            for X, y in train_loader:
                optimizer.zero_grad()
                modelo.train()
                X, y = X.to(device), y.to(device)
                pred = modelo(X)
                if pred.shape != y.shape:
                    raise ValueError(f"forma de la predicción {tuple(pred.shape)} != objetivo {tuple(y.shape)}")
                loss = criterion(pred, y, mode="train")
                conteo[criterion.ultimo_termino] += 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=hp["max_norm"])
                optimizer.step()
                suma += loss.detach(); n_b += 1
                if n_b % CHEQUEO_NAN_CADA == 0 and not torch.isfinite(suma).item():
                    raise PerdidaNoFinita(epoch + 1, n_b)
            train_loss = (suma / max(n_b, 1)).item()
            if not np.isfinite(train_loss):
                raise PerdidaNoFinita(epoch + 1, n_b)

            val, v_fal, v_fcl, v_wmse, met = validar(modelo, criterion, val_loader)
            if not np.isfinite(val):
                trial.set_user_attr("nan_val_en_epoca", epoch + 1)
                print(f"[Trial {trial.number}] val no finito ({val}): podado.", flush=True)
                raise optuna.TrialPruned()

            scheduler.step()                           # una vez por época, después de validar (como objetivo_b.py)

            if val < mejor_val:
                mejor_val, mejor_ep = val, epoch + 1
                _guardar_atomico({"modelo": _pesos_cpu(modelo), "epoca": epoch + 1, "val": val, "val_fal": v_fal,
                                  "val_fcl": v_fcl, "val_wmse": v_wmse, **met}, r_best)
            hist.append(dict(epoca=epoch + 1, train_loss=train_loss, val_loss=val, val_fal=v_fal, val_fcl=v_fcl,
                             val_wmse=v_wmse, q_wmse=criterion.prob_wmse(), n_wmse=conteo["WMSE"],
                             n_fal=conteo["FAL"], n_fcl=conteo["FCL"], lr=lr_ep,
                             seg_epoca=time.time() - t_ep, **met))
            trial.report(val, epoch)

            _guardar_atomico({
                "hparams": hp_id, "modelo": modelo.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "criterion_step": criterion.step,
                "py_rng": random.getstate(), "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                "gen_state": gen.get_state(), "epoca": epoch + 1, "mejor_val": mejor_val, "mejor_epoca": mejor_ep,
                "historial": hist, "tiempo_acum": t_prev + (time.time() - t_inicio)}, r_resume)
            _escribir_curva(r_curva, hist)
            print(f"[Trial {trial.number}] Ep {epoch+1}/{epocas} | train={train_loss:.4g} | val={val:.5f} "
                  f"(WMSE {v_wmse:.3g} FAL {v_fal:.2e} FCL {v_fcl:.4f}) | CSI={met['csi_media']:.4f} "
                  f"FSS={met[FSS_REF]:.4f} | Q={criterion.prob_wmse():.2f} | {time.time()-t_ep:.0f}s", flush=True)

            if trial.should_prune():
                raise optuna.TrialPruned()
            if paciencia and len(hist) - mejor_ep >= paciencia:
                print(f"[Trial {trial.number}] early stopping en la época {len(hist)} (mejor: {mejor_ep}).", flush=True)
                trial.set_user_attr("early_stopping_en_epoca", len(hist))
                break

        ult = hist[-1]
        _guardar_atomico({"modelo": _pesos_cpu(modelo), "epoca": len(hist), "val": ult["val_loss"], "val_fal": ult["val_fal"],
                          "val_fcl": ult["val_fcl"], "val_wmse": ult["val_wmse"],
                          **{c: ult[c] for c in columnas_metricas()}}, r_last)
        _escribir_curva(r_curva, hist)
        trial.set_user_attr("tiempo_trial_seg", t_prev + (time.time() - t_inicio))
        trial.set_user_attr("mejor_epoca", mejor_ep)
        trial.set_user_attr("val_final", ult["val_loss"])
        trial.set_user_attr("criterio_val", "(WMSE + sqrt(HW)*FAL + sqrt(HW)*FCL)/3")
        csis = [(h["csi_media"], h["epoca"]) for h in hist if np.isfinite(h["csi_media"])]
        fsss = [(h[FSS_REF], h["epoca"]) for h in hist if np.isfinite(h[FSS_REF])]
        if csis:
            trial.set_user_attr("csi_media_mejor", max(csis)[0]); trial.set_user_attr("csi_media_mejor_epoca", max(csis)[1])
        if fsss:
            trial.set_user_attr("fss_u15_v3_mejor", max(fsss)[0]); trial.set_user_attr("fss_u15_v3_mejor_epoca", max(fsss)[1])
        return mejor_val

    except optuna.TrialPruned:
        raise
    except PerdidaNoFinita as e:
        trial.set_user_attr("nan_en_epoca", e.epoca); trial.set_user_attr("nan_en_lote", e.lote)
        print(f"[Trial {trial.number}] {e}: podado.", flush=True)
        raise optuna.TrialPruned()
    except Exception as e:
        _vaciar_cache()
        if _es_oom(e):
            trial.set_user_attr("oom", True)
            print(f"[Trial {trial.number}] OOM con batch_size={batch_size}: podado.", flush=True)
            raise optuna.TrialPruned()
        print(f"[Trial {trial.number}] ERROR {type(e).__name__}: {e}", flush=True)
        raise


def objective_A(trial):
    n = trial.suggest_categorical("n_niveles", NIVELES_A)
    f = trial.suggest_categorical("n_filtros_base", FILTROS_A)
    ks = [trial.suggest_categorical(f"kernel_{i}", KERNELS_A) for i in range(n)]
    ps = [trial.suggest_float(f"dropout_{i}", *DROPOUT_A) for i in range(n)]
    arq = dict(n_niveles=n, n_filtros_base=f, kernel_sizes=ks, dropout_ps=ps)
    dir_id = int(trial.user_attrs.get("resume_from", trial.number))
    hp = dict(FIJOS_A)
    return entrenar(trial, os.path.join(DIR, f"trial_{dir_id:04d}"), arq, hp)


def objective_B(trial):
    hp = dict(FIJOS_B)
    hp["lr"] = trial.suggest_float("lr", *LR_RANGO, log=True)
    hp["batch_size"] = trial.suggest_categorical("batch_size", BATCH_B)
    hp["max_prob"] = trial.suggest_float("max_prob", *MAX_PROB_RANGO)
    hp["max_norm"] = trial.suggest_float("max_norm", *MAX_NORM_RANGO, log=True)
    dir_id = int(trial.user_attrs.get("resume_from", trial.number))
    return entrenar(trial, os.path.join(DIR, f"trial_{dir_id:04d}"), ARQ_B, hp)


objective = objective_A if ETAPA == "A" else objective_B
