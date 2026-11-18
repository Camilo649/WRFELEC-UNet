# =====================================================================
# CHECKPOINT DE LA ETAPA B OFICIAL — ensayo 100, mejor época (best.pt)
# Es el modelo final de la monografía (Temp Ext Clip U-Net FACL).
# Arma un `checkpoint` con el mismo formato que el de temporal_*_best.pth
# para que sigan andando resumen_entrenamiento, graficar_curva_error,
# graficar_reconstruccion, etc.
# OJO: acá val_loss es el MAE normalizado (criterio oficial), no la FACL.
# Archivos en Serafin:
#   ../estudios/optuna_etapa_B_oficial.log
#   ../modelos/etapa_B_oficial/trial_0100/{best.pt,curva.csv}
# =====================================================================
import optuna
import pandas as pd
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

TRIAL = 100
CKPT = 'best.pt'     # también: 'best_csi.pt', 'best_fss.pt', 'last.pt'
DIR_TRIAL = f'../modelos/etapa_B_oficial/trial_{TRIAL:04d}'
BATCH_VAL_B = 16     # lote de validación de la Etapa B oficial (el MAE
                     # oficial se promedia por lote)

# Hiperparámetros del ensayo: se leen del estudio (no se copian a mano)
est = optuna.load_study(
    study_name="etapa_B_Memo_Fat_Short_Unet_FACL_oficial",
    storage=JournalStorage(
        JournalFileBackend("../estudios/optuna_etapa_B_oficial.log")))
t = next(x for x in est.trials if x.number == TRIAL)
HP = dict(lr=t.params['lr'], alpha=t.params['alpha'],
          epocas=t.params['epocas'], batch_size=t.params['batch_size'])
N_PARAMETROS = t.user_attrs['n_parametros']

# Arquitectura fija de la Etapa B oficial (ARQ_FINAL de
# objetivo_B_oficial.py: trial #118 de la Etapa A oficial). NO es la
# de la FACL propia (64 filtros, kernels [9,5,3]).
ARQ_B = dict(n_niveles=3, n_filtros_base=32, kernel_sizes=[5, 5, 5],
             dropout_ps=[0.08, 0.06, 0.20])

best = torch.load(f'{DIR_TRIAL}/{CKPT}', map_location=device,
                  weights_only=False)      # checkpoint propio, de confianza
curva = pd.read_csv(f'{DIR_TRIAL}/curva.csv')

# Sanidad: la época y el MAE del .pt coinciden con curva.csv
fila = curva[curva['epoca'] == int(best['epoca'])].iloc[0]
assert abs(best['val'] - fila['val_loss']) < 1e-6, (best['val'], fila['val_loss'])
if CKPT == 'best.pt':
    assert int(best['epoca']) == int(curva['epoca'].iloc[curva['val_loss'].idxmin()])
assert len(curva) == HP['epocas'], f"curva.csv tiene {len(curva)} épocas"

checkpoint = dict(
    model_state_dict=best['modelo'],
    arquitectura=ARQ_B,
    in_channels=N_CANALES,
    nombres_canales=NOMBRES_CANALES,
    batch_size=HP['batch_size'],
    lr=HP['lr'],
    epochs_entrenadas=len(curva),
    historial_train_loss=curva['train_loss'].tolist(),
    historial_val_loss=curva['val_loss'].tolist(),
    historial_tiempo_epoch=curva['seg_epoca'].tolist(),
    mejor_val_loss=float(curva['val_loss'].min()),
    mejor_epoca=int(best['epoca']),
    patience='sin early stopping',
)

# Loaders: batch del ensayo para train (8) y 16 para validación
train_loader = DataLoader(train_dataset, batch_size=HP['batch_size'],
                          shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_VAL_B,
                          shuffle=False)

modelo = UNet_WRF(in_channels=checkpoint['in_channels'],
                  **checkpoint['arquitectura']).to(device)
modelo.load_state_dict(checkpoint['model_state_dict'])
modelo.eval()

n_par = sum(p.numel() for p in modelo.parameters())
assert n_par == N_PARAMETROS, f"{n_par:,} parámetros (esperados {N_PARAMETROS:,})"
assert n_par == 5_410_241, f"no es el modelo final: {n_par:,} parámetros"
print(f"Ensayo {TRIAL} ({CKPT}): {n_par:,} parámetros, época "
      f"{checkpoint['mejor_epoca']} de {HP['epocas']} | MAE val "
      f"{best['val']:.6f} | lr={HP['lr']:.2e}, alpha={HP['alpha']:.4f}")

assert checkpoint['epochs_entrenadas'] == HP['epocas']
resumen_entrenamiento(checkpoint, NOMBRE_MODELO, train_loader, val_loader,
                      checkpoint['batch_size'])
