# -*- coding: utf-8 -*-
# =====================================================================
# CELDA PARA Entrenamiento_FatShortUnetFACL.ipynb  (se puede correr
# en el JupyterHub de Serafín o en Clementina; hace falta una GPU)
#
# Qué hace: obtiene los pesos de la Ext Clip U-Net FACL en su "mejor
# época según L_val" (la 8), que la reproducción de las curvas no
# guardó, para mostrar en Resultados cómo cambia la serie temporal de
# validación según qué pérdida de validación elige la época.
#   PARTE 1: reentrena el ensayo 39 de la Fase 2 con exactamente lo
#            mismo que celda_curvas_fal_fcl_trial39.py (semillas,
#            hiperparámetros leídos de Optuna, arquitectura del
#            ensayo 106), pero SOLO LAS PRIMERAS 8 ÉPOCAS, con el
#            cronograma de las 140 (T_max, FACL y probabilidad de
#            usar la FCL), y guarda los pesos de la época 8.
#            En Clementina dura ~1 h 10 min (8 épocas de ~8,3 min);
#            en el JupyterHub tardará más (la Ext Clip U-Net Comb
#            V2.1, de 8 millones de parámetros, tardó ~25 min por
#            época ahí), así que contá con unas 3 o 4 horas.
#   PARTE 2: estima con ese modelo las 2.271 muestras de validación y
#            guarda resultados_validacion_epoca8.npz (el archivo
#            anterior más la clave `pred_ext_epoca8`).
#
# Antes de correrla, ejecutá las celdas de: Dependencias, semillas
# (SEED = 42), UNet_WRF, FACL, DATASETS (incluye TormentaDataset) e
# `import optuna`. Copiá resultados_validacion.npz a la carpeta
# desde la que corrés (o cambiá RUTA_NPZ). Si se corta, volvé a
# correrla: sigue desde la última época completa.
#
# Salidas: ../modelos/etapa_B_Fat_Short_Unet_FACL/
#              B_trial0039_repro_epoca8.pth
#          resultados_validacion_epoca8.npz   <- mandame este
# =====================================================================
import glob
import json
import os
import time

TRIAL_ARQ, TRIAL_ENT = 106, 39
BATCH_VAL = 16
EPOCA_FINAL = 8      # se entrena con el cronograma de 140 épocas
SALIDA = f'../estudios/trial{TRIAL_ENT:04d}_hasta_epoca8.json'
CKPT = (f'../modelos/etapa_B_Fat_Short_Unet_FACL/'
        f'B_trial{TRIAL_ENT:04d}_repro_epoca8.pth')


def buscar_estudio(patron):
    """Primer estudio de Optuna cuyo .db coincide con el patrón."""
    for db in sorted(glob.glob(f'../estudios/{patron}')):
        storage = f'sqlite:///{db}'
        for resumen in optuna.get_all_study_summaries(storage):
            print(f'  {os.path.basename(db)} -> {resumen.study_name}')
            return optuna.load_study(study_name=resumen.study_name,
                                     storage=storage)
    raise FileNotFoundError(f'No encontré ../estudios/{patron}')


print('Estudios encontrados:')
est_arq = buscar_estudio('optuna_arquitectura_*Fat Short Unet FACL*.db')
est_ent = buscar_estudio('optuna_entrenamiento_*Fat Short Unet FACL*.db')
pa = est_arq.trials[TRIAL_ARQ].params
pe = est_ent.trials[TRIAL_ENT].params
n_niv = pa['n_niveles']
ARQ = dict(n_niveles=n_niv, n_filtros_base=pa['n_filtros_base'],
           kernel_sizes=[pa[f'kernel_nivel_{i}'] for i in range(n_niv)],
           dropout_ps=[pa[f'dropout_nivel_{i}'] for i in range(n_niv)])
lr, alpha = pe['lr'], pe['alpha']
epocas, batch_size = pe['epocas'], pe['batch_size']
print('Arquitectura (Fase 1, ensayo 106):', ARQ)
print(f'Entrenamiento (Fase 2, ensayo 39): lr={lr}, alpha={alpha}, '
      f'épocas={epocas}, batch={batch_size}')
print('Atributos guardados del ensayo 39:',
      est_ent.trials[TRIAL_ENT].user_attrs)

# ---- Igual que objective_entrenamiento, en el mismo orden ----------
torch.cuda.empty_cache()
generator = torch.Generator()
generator.manual_seed(SEED)
train_loader_r = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, generator=generator,
                            num_workers=0, pin_memory=True)
val_loader_r = DataLoader(val_dataset, batch_size=BATCH_VAL,
                          shuffle=False, num_workers=0, pin_memory=True)
torch.manual_seed(SEED + TRIAL_ENT)
np.random.seed(SEED + TRIAL_ENT)
modelo_r = UNet_WRF(in_channels=3, **ARQ).to(device)
print('Parámetros:', sum(p.numel() for p in modelo_r.parameters()),
      '(tiene que dar 9.555.457)')
criterion_r = FACL(total_steps=epocas * len(train_loader_r),
                   alpha=alpha).to(device)
optimizer_r = optim.Adam(modelo_r.parameters(), lr=lr)
scheduler_r = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_r, T_max=epocas, eta_min=1e-7)

hist = dict(epoca=[], L_val=[], FAL=[], FCL=[], prob_fcl=[],
            frac_pasos_fcl=[], lr=[], segundos=[])
inicio = 0
if os.path.exists(CKPT):                      # reanudar
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    modelo_r.load_state_dict(ck['modelo'])
    optimizer_r.load_state_dict(ck['optimizer'])
    scheduler_r.load_state_dict(ck['scheduler'])
    criterion_r.step = ck['criterion_step']
    # map_location=device mueve los estados del RNG a la GPU, y
    # set_rng_state exige ByteTensor en la CPU: se los devuelve ahí.
    torch.set_rng_state(ck['rng_torch'].cpu())
    torch.cuda.set_rng_state_all([t.cpu() for t in ck['rng_cuda']])
    np.random.set_state(ck['rng_numpy'])
    generator.set_state(ck['rng_loader'].cpu())
    hist, inicio = ck['hist'], ck['epoca']
    print(f'Reanudando desde la época {inicio + 1}')

for epoch in range(inicio, EPOCA_FINAL):
    t0 = time.time()
    lr_epoca = optimizer_r.param_groups[0]['lr']
    modelo_r.train()
    n_fcl = n_tot = 0
    for batch_X, batch_y in tqdm(train_loader_r, leave=False,
                                 desc=f'Época {epoch + 1}/{epocas}'):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer_r.zero_grad()
        loss = criterion_r(modelo_r(batch_X), batch_y, mode='train')
        loss.backward()
        optimizer_r.step()
        n_tot += 1
    # Fracción esperada de pasos con FCL en la época (P(t) promedio)
    paso_fin = criterion_r.step
    frac = np.mean([criterion_r.prob_fcl(t)
                    for t in range(paso_fin - n_tot, paso_fin)])

    modelo_r.eval()
    criterion_r.val_reset()
    with torch.no_grad():
        for batch_X, batch_y in val_loader_r:
            criterion_r.val_acumular(modelo_r(batch_X.to(device)),
                                     batch_y.to(device))
    lval = criterion_r.val_resultado()
    scheduler_r.step()

    for k, v in (('epoca', epoch + 1), ('L_val', lval),
                 ('FAL', criterion_r.val_fal),
                 ('FCL', criterion_r.val_fcl),
                 ('prob_fcl', criterion_r.prob_fcl()),
                 ('frac_pasos_fcl', float(frac)), ('lr', lr_epoca),
                 ('segundos', time.time() - t0)):
        hist[k].append(float(v))
    print(f'Época {epoch + 1}/{epocas} | L_val={lval:.6f} | '
          f'FAL={criterion_r.val_fal:.6f} FCL={criterion_r.val_fcl:.6f}'
          f' | P(FCL)={criterion_r.prob_fcl():.3f} | '
          f'{time.time() - t0:.0f}s')

    with open(SALIDA, 'w', encoding='utf-8') as f:
        json.dump(dict(arquitectura=ARQ, lr=lr, alpha=alpha,
                       epocas=epocas, batch_size=batch_size,
                       hist=hist), f, indent=1)
    torch.save(dict(modelo=modelo_r.state_dict(),
                    optimizer=optimizer_r.state_dict(),
                    scheduler=scheduler_r.state_dict(),
                    criterion_step=criterion_r.step,
                    rng_torch=torch.get_rng_state(),
                    rng_cuda=torch.cuda.get_rng_state_all(),
                    rng_numpy=np.random.get_state(),
                    rng_loader=generator.get_state(),
                    hist=hist, epoca=epoch + 1), CKPT)

print('\nControl: época 8  L_val =', round(hist['L_val'][-1], 6),
      '(en la reproducción de las curvas: 0,046700; tiene que ser',
      'el mínimo de L_val)')
print('         mínimo de L_val hasta ahora en la época',
      hist['epoca'][int(np.argmin(hist['L_val']))], '(esperada: 8)')
print(f'Guardado {SALIDA} y {CKPT} (pesos de la época 8)')

# =====================================================================
# PARTE 2: estimaciones del modelo de la época 8 sobre la validación
# Hace falta resultados_validacion.npz (el que ya generó
# celda_resultados_validacion.py) en la carpeta actual. Agrega la
# clave `pred_ext_epoca8` y guarda resultados_validacion_epoca8.npz.
# =====================================================================
DIR_EXT = '../dataset'          # conjunto extendido, 3 canales
RUTA_NPZ = 'resultados_validacion.npz'
RUTA_SALIDA = 'resultados_validacion_epoca8.npz'
LOTE, TAM = 16, 297

ck8 = torch.load(CKPT, map_location=device, weights_only=False)
assert ck8['epoca'] == EPOCA_FINAL, ck8['epoca']
m8 = UNet_WRF(in_channels=3, **ARQ).to(device)
m8.load_state_dict(ck8['modelo'])
m8.eval()

previo = dict(np.load(RUTA_NPZ))
nombres = [str(n) for n in previo['archivos']]
ds_e = TormentaDataset(split_dir=f'{DIR_EXT}/val',
                       stats_path=f'{DIR_EXT}/stats.npz')
idx_e = {f.name: i for i, f in enumerate(ds_e.files)}
assert all(n in idx_e for n in nombres), 'faltan muestras'

out = np.zeros((len(nombres), TAM, TAM), dtype=np.uint8)
with torch.no_grad():
    for a in tqdm(range(0, len(nombres), LOTE), desc='época 8'):
        lote = nombres[a:a + LOTE]
        X = torch.stack([ds_e[idx_e[n]][0] for n in lote]).to(device)
        p = m8(X).cpu().squeeze(1).clamp(0, 1)
        for k in range(len(lote)):
            v = np.round(ds_e.desnormalizar_y(p[k]).numpy())
            assert v.min() >= 0 and v.max() <= 255
            out[a + k] = v.astype(np.uint8)
previo['pred_ext_epoca8'] = out

real = previo['y_real']
act = lambda M: M.reshape(len(M), -1).sum(1, dtype=np.int64)
print('Cociente estimada/simulada (todo el conjunto):')
for k in ('pred_ext_ultima', 'pred_ext_epoca8', 'pred_comb_mejor'):
    print(f'  {k:18s}: {act(previo[k]).sum() / act(real).sum():.2f}')
np.savez_compressed(RUTA_SALIDA, **previo)
print(f'Guardado {RUTA_SALIDA} '
      f'({os.path.getsize(RUTA_SALIDA) / 1e6:.1f} MB). '
      'Mandame ese archivo y esta salida.')
