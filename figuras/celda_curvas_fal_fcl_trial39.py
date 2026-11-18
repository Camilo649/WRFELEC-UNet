# -*- coding: utf-8 -*-
# =====================================================================
# CELDA PARA Entrenamiento_Ext_Clip_U-Net_FACL.ipynb
#
# Qué hace: vuelve a entrenar el ensayo 39 de la Fase 2 (la Ext Clip
# U-Net FACL) exactamente como lo hizo `objective_entrenamiento`
# (mismas semillas, mismos hiperparámetros, misma arquitectura), pero
# guardando en cada época la FAL, la FCL y L_val de validación. Con
# eso armo las curvas de los dos términos por separado.
#
# Los hiperparámetros NO están escritos a mano: se leen de los
# estudios de Optuna (ensayo 106 de la Fase 1 para la arquitectura y
# ensayo 39 de la Fase 2 para el entrenamiento), para que coincidan
# con los originales hasta el último decimal.
#
# Antes de correrla, ejecutá las celdas de: Dependencias, semillas
# (SEED = 42), UNet_WRF, FACL, DATASETS e `import optuna`.
# No hace falta la celda de HIPERPARÁMETROS.
#
# Dura lo mismo que el ensayo original (unas 19 h). Guarda todo al
# final de cada época, así que si se corta, volvé a correrla y sigue
# desde la última época completa.
#
# Control: en la época 5 tiene que dar L_val ≈ 0,046355 y en la 140,
# L_val ≈ 0,055695 (lo que midió la celda de la importancia). Si da
# muy distinto, avisame antes de usar las curvas.
#
# Salida: ../estudios/curvas_fal_fcl_trial0039.json  <- mandame este
#         (se actualiza en cada época; los pesos van aparte, en
#         ../modelos/etapa_B_Fat_Short_Unet_FACL/, y no pisan nada)
# =====================================================================
import glob
import json
import os
import time

TRIAL_ARQ, TRIAL_ENT = 106, 39
BATCH_VAL = 16
SALIDA = f'../estudios/curvas_fal_fcl_trial{TRIAL_ENT:04d}.json'
CKPT = (f'../modelos/etapa_B_Fat_Short_Unet_FACL/'
        f'B_trial{TRIAL_ENT:04d}_repro_curvas.pth')


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
    torch.set_rng_state(ck['rng_torch'])
    torch.cuda.set_rng_state_all(ck['rng_cuda'])
    np.random.set_state(ck['rng_numpy'])
    generator.set_state(ck['rng_loader'])
    hist, inicio = ck['hist'], ck['epoca']
    print(f'Reanudando desde la época {inicio + 1}')

for epoch in range(inicio, epocas):
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

i5 = hist['epoca'].index(5) if 5 in hist['epoca'] else None
print('\nControl: época 5  L_val =',
      None if i5 is None else round(hist['L_val'][i5], 6),
      '(original 0,046355)')
print('         última    L_val =', round(hist['L_val'][-1], 6),
      '(original 0,055695)')
print('Mejor época de la reproducción:',
      hist['epoca'][int(np.argmin(hist['L_val']))])
print(f'Guardado {SALIDA}')
