#!/bin/bash
#SBATCH --job-name=mejor_ofi_200
#SBATCH --partition=gpunode
#SBATCH --nodes=1
#SBATCH --gres=gpu:intel_xt1550:1
#SBATCH --cpus-per-task=8
#SBATCH --time=30:00:00
#SBATCH --exclude=cn045,cn072,cn021
#SBATCH --output=logs/mejor_oficial_200_%j.log

# Reentrena la configuración del mejor trial de la Etapa B OFICIAL (lr, alpha, batch_size) con más épocas.
# Lanza N_RUNS corridas en paralelo, una por tile:
#   DIR_ID = TRIAL_BASE      -> misma semilla que el trial original (solo cambian las épocas)
#   DIR_ID = TRIAL_BASE + k  -> otra semilla (control: cuánto varía el resultado solo por la semilla)
# Uso:  sbatch sbatch_mejor_oficial_200.sh
#       sbatch --export=ALL,TRIAL_BASE=100,EPOCAS=200,N_RUNS=2 sbatch_mejor_oficial_200.sh
# Si Slurm lo corta, se vuelve a lanzar igual y reanuda desde resume.pt.

source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch_xpu
cd ~/lightning
mkdir -p logs estudios
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1

unset SUBSET SMOKE DEVICE STUDY_NAME JOURNAL DIR_B N_STARTUP EPOCAS_TRIAL_A WARMUP_FRAC LR N_WORKERS TEST_BATCH
export DATASET_DIR=$HOME/data/dataset_Camilo_temporal    # dataset TEMPORAL (9 canales)
export TRIAL_BASE=${TRIAL_BASE:-100}
export EPOCAS=${EPOCAS:-200}
N_RUNS=${N_RUNS:-2}

echo "job $SLURM_JOB_ID | nodo $(hostname) | trial base $TRIAL_BASE | épocas $EPOCAS | corridas $N_RUNS"

NDEV=$(timeout -k 10 180 python -c "import torch; print(torch.xpu.device_count())" 2>/dev/null | tail -n 1)
echo "device_count: ${NDEV:-<sin respuesta>} (se requieren >= $N_RUNS)"
if ! [[ "$NDEV" =~ ^[0-9]+$ ]] || [ "$NDEV" -lt "$N_RUNS" ]; then
    echo "ERROR: dispositivos XPU insuficientes en $(hostname); se aborta." >&2
    exit 1
fi

for k in $(seq 0 $((N_RUNS-1))); do
    ZE_AFFINITY_MASK=$k OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/N_RUNS)) DIR_ID=$((TRIAL_BASE+k)) \
        python run_mejor_200.py > logs/mejor_oficial_${EPOCAS}_${SLURM_JOB_ID}_id$((TRIAL_BASE+k)).log 2>&1 &
done
wait
