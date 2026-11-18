#!/bin/bash
#SBATCH --job-name=prueba_A_oficial
#SBATCH --partition=testing
#SBATCH --nodelist=cn034
#SBATCH --gres=gpu:intel_xt1550:4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:20:00
#SBATCH --output=logs/prueba_A_oficial_%j.log

# Prueba de la Etapa A con la FACL oficial en la partición testing (límite 20 min): los mismos pasos que
# optuna_A_oficial.sbatch, con 8 workers y trials de 2 épocas sobre un subconjunto chico, en un estudio y
# un journal APARTE (no toca el de la corrida real).
# Uso:
#   sbatch sbatch_prueba_A_oficial.sh                  # lr oficial (1e-3)
#   LR=2.5e-5 sbatch sbatch_prueba_A_oficial.sh        # diagnóstico: misma FACL oficial, otro lr

source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch_xpu
cd ~/lightning
mkdir -p logs estudios
export PYTHONPATH=$PWD
unset DEVICE N_WORKERS N_STARTUP

export LR=${LR:-1e-3}
export DATASET_DIR=$HOME/data/dataset_Camilo_temporal   # dataset TEMPORAL (9 canales)
export SUBSET=512                                       # 512 muestras de train y 128 de val: trials de segundos
export STUDY_NAME=TEST_A_oficial_lr$LR JOURNAL=estudios/TEST_A_oficial_lr$LR.log
export EPOCAS_TRIAL_A=2 WARMUP_FRAC=0 TIMEOUT_S=480     # ~8 min
echo "PRUEBA con lr=$LR | estudio=$STUDY_NAME"

# Chequeo: el dataset tiene que tener 9 canales
python - <<'PYEOF' || exit 1
import os, numpy as np
d = os.environ["DATASET_DIR"]
n = int(np.load(os.path.join(d, "stats.npz"))["min_X"].shape[0])
print(f"DATASET_DIR={d} | canales={n}")
assert n == 9, f"se esperaban 9 canales y hay {n}"
PYEOF

MIN_DEV=${MIN_DEV:-8}
NDEV=$(timeout -k 10 180 python -c "import torch; print(torch.xpu.device_count())" 2>/dev/null | tail -n 1)
echo "device_count: ${NDEV:-<sin respuesta>} (se requieren >= $MIN_DEV) | nodo: $(hostname)"
if ! [[ "$NDEV" =~ ^[0-9]+$ ]] || [ "$NDEV" -lt "$MIN_DEV" ]; then
    echo "ERROR: dispositivos XPU insuficientes en $(hostname); se aborta el job." >&2
    exit 1
fi

ZE_AFFINITY_MASK=0 timeout -k 10 600 python test_facl_oficial.py || { echo "ERROR: falló test_facl_oficial.py; se aborta el job." >&2; exit 1; }

export N_WORKERS=$NDEV
export N_STARTUP=$((2*NDEV))
echo "workers: $N_WORKERS | n_startup_trials (sampler y pruner): $N_STARTUP"

for i in $(seq 0 $((NDEV-1))); do
    ZE_AFFINITY_MASK=$i OMP_NUM_THREADS=$((64/NDEV)) \
        python worker_A_oficial.py $i > logs/prueba_A_oficial_${SLURM_JOB_ID}_w$i.log 2>&1 &
done
wait
