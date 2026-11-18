# -*- coding: utf-8 -*-
# =====================================================================
# CELDA PARA Entrenamiento_Ext_Clip_U-Net_FACL.ipynb  (versión 2)
#
# Qué hace: corre la importancia por permutación (100 repeticiones
# por variable) con los pesos de la última época (la 140) del ensayo
# 39 reproducido con `celda_curvas_fal_fcl_trial39.py`, y verifica
# que su L_val sea ≈ 0,0557.
#
# Arma ella misma el modelo con la arquitectura del ensayo 106, que
# está escrita en esta celda. NO usa el `modelo` que deja la celda de
# HIPERPARÁMETROS.
#
# Antes de correrla, ejecutá las celdas de: Dependencias, UNet_WRF,
# FACL, DATASETS e importancia (cargar_set_en_memoria y
# permutation_importance_global). No hace falta la de las curvas.
#
# Salida: ../estudios/pfi_repro_epoca140_resumen.txt  <- mandame este
# =====================================================================
import json
import os

CKPT = ('../modelos/etapa_B_Fat_Short_Unet_FACL/'
        'B_trial0039_repro_curvas.pth')
N_REPEATS = 100
nombres_wrf = ['MDBZ', 'HIELO', 'CTT']

# Arquitectura (ensayo 106 de la Fase 1) e hiperparámetros del ensayo
# 39 de la Fase 2, tal como quedaron en curvas_fal_fcl_trial0039.json
ARQ = dict(n_niveles=3, n_filtros_base=64, kernel_sizes=[5, 5, 3],
           dropout_ps=[0.024461173266027245, 0.07965792664013316,
                       0.27383264907951727])
alpha, epocas = 0.18908467719664426, 140
resumen = []


def anotar(texto):
    print(texto)
    resumen.append(texto)


ck = torch.load(CKPT, map_location=device, weights_only=False)
anotar(f"Checkpoint de la reproducción: época {ck['epoca']}")
assert ck['epoca'] == 140, 'el entrenamiento no terminó'
modelo_p = UNet_WRF(in_channels=3, **ARQ).to(device)
modelo_p.load_state_dict(ck['modelo'])
modelo_p.eval()
criterion_p = FACL(total_steps=epocas * 1125, alpha=alpha).to(device)

if 'X_val' not in globals():          # se carga una sola vez
    X_val, Y_val = cargar_set_en_memoria(
        DataLoader(val_dataset, batch_size=16, shuffle=False))

# L_val y términos con los pesos de la última época
criterion_p.val_reset()
with torch.no_grad():
    for i in range(0, X_val.shape[0], 8):
        criterion_p.val_acumular(
            modelo_p(X_val[i:i + 8].to(device)),
            Y_val[i:i + 8].to(device))
lval = criterion_p.val_resultado()
anotar(f"ultima  L_val={lval:.6f}  FAL={criterion_p.val_fal:.6f}  "
       f"FCL={criterion_p.val_fcl:.6f}  (esperado ≈ 0,055700)")

# Importancia por permutación
df_imp, base_loss = permutation_importance_global(
    modelo_p, X_val, Y_val, criterion_p, nombres_wrf,
    n_repeats=N_REPEATS, batch_size=8, device=device, seed=SEED)
anotar(f"Baseline (época 140): {base_loss:.6f}")
anotar(df_imp.to_string(index=False))
for _, f in df_imp.iterrows():
    anotar(f"{f['feature']:6s} aumento "
           f"{100 * f['importance_mean'] / base_loss:7.1f} % "
           f"(desvío {100 * f['importance_std'] / base_loss:.2f} %)")
anotar('JSON: ' + json.dumps(dict(
    baseline=base_loss,
    importancia={r['feature']: [r['importance_mean'],
                                r['importance_std'],
                                int(r['n_repeats'])]
                 for _, r in df_imp.iterrows()})))
os.makedirs('../estudios', exist_ok=True)
with open('../estudios/pfi_repro_epoca140_resumen.txt', 'w',
          encoding='utf-8') as f:
    f.write('\n'.join(resumen))
print('\nGuardado ../estudios/pfi_repro_epoca140_resumen.txt')
