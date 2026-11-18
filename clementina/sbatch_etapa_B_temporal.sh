#!/bin/bash
#SBATCH --job-name=optuna_B_temp
#SBATCH --partition=gpunode
#SBATCH --nodes=1
#SBATCH --gres=gpu:intel_xt1550:4
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --exclusive                     # el nodo entero (CPU, RAM y las 8 tiles), sin compartir con otros jobs
#SBATCH --exclude=cn045,cn072,cn021    # cn045: XPU defectuoso; cn072 y cn021: el test_B_* se colgó ahí el 26/10 (1551945/46/63)
#SBATCH --output=logs/optuna_B_temporal_%j.log

# Etapa B TEMPORAL (dataset de 9 canales, arquitectura del trial #56, todas las épocas guardadas, CSI y FSS por época).
# Uso (dos jobs en paralelo sobre el mismo estudio, como la Etapa B anterior):
#   sbatch --export=ALL,JOB_INDEX=0 sbatch_etapa_B_temporal.sh
#   sbatch --export=ALL,JOB_INDEX=1 sbatch_etapa_B_temporal.sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch_xpu
cd ~/lightning
mkdir -p logs estudios etapa_B_temporal
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1

# que no se hereden variables de las pruebas
unset SUBSET SMOKE DEVICE STUDY_NAME JOURNAL DIR_B N_STARTUP EPOCAS_TRIAL_A WARMUP_FRAC LR N_WORKERS TEST_BATCH

export JOB_INDEX=${JOB_INDEX:-0}
if ! [[ "$JOB_INDEX" =~ ^[01]$ ]]; then echo "ERROR: JOB_INDEX tiene que ser 0 o 1 (es '$JOB_INDEX')" >&2; exit 1; fi
export DATASET_DIR=$HOME/data/dataset_Camilo_temporal   # dataset TEMPORAL (9 canales)
export TIMEOUT_S=$((46*3600))                           # no lanza trials nuevos pasadas 46 h
export N_STARTUP=16                                     # trials aleatorios antes de que arranque el TPE

echo "job $SLURM_JOB_ID | JOB_INDEX=$JOB_INDEX | nodo $(hostname) | N_STARTUP=$N_STARTUP"

# Chequeo: el dataset tiene que tener 9 canales
python - <<'PYEOF' || exit 1
import os, numpy as np
d = os.environ["DATASET_DIR"]
n = int(np.load(os.path.join(d, "stats.npz"))["min_X"].shape[0])
print(f"DATASET_DIR={d} | canales={n}")
assert n == 9, f"se esperaban 9 canales y hay {n}"
PYEOF

# Cantidad de dispositivos XPU, con timeout (en cn045 este comando quedó colgado 7 h al terminar)
MIN_DEV=${MIN_DEV:-8}
NDEV=$(timeout -k 10 180 python -c "import torch; print(torch.xpu.device_count())" 2>/dev/null | tail -n 1)
echo "device_count: ${NDEV:-<sin respuesta>} (se requieren >= $MIN_DEV) | nodo: $(hostname)"
if ! [[ "$NDEV" =~ ^[0-9]+$ ]] || [ "$NDEV" -lt "$MIN_DEV" ]; then
    echo "ERROR: dispositivos XPU insuficientes en $(hostname); se aborta el job." >&2
    exit 1
fi

# Batch 32 con la arquitectura final tiene que entrar en memoria, y CSI/FSS tienen que dar igual que en CPU
ZE_AFFINITY_MASK=0 timeout -k 10 900 python test_B_temporal.py
rc=$?
if [ $rc -ne 0 ]; then
    echo "ERROR: falló test_B_temporal.py (código $rc; 124 = timeout, se colgó); se aborta el job." >&2
    echo "--- procesos más activos en $(hostname) ---"; ps -eo pid,user,etime,pcpu,pmem,cmd --sort=-pcpu | head -15
    exit 1
fi

for i in $(seq 0 $((NDEV-1))); do
    ZE_AFFINITY_MASK=$i OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/NDEV)) \
        python worker_B_temporal.py $i > logs/B_temporal_j${JOB_INDEX}_${SLURM_JOB_ID}_w$i.log 2>&1 &
done
wait
