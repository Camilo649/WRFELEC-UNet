#!/bin/bash
#SBATCH --job-name=optuna_B_comb
#SBATCH --partition=gpunode
#SBATCH --nodes=1
#SBATCH --gres=gpu:intel_xt1550:4
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --exclusive
#SBATCH --exclude=cn045,cn072,cn021,cn033
#SBATCH --output=logs/optuna_B_comb_%j.log

# Etapa B (Fase 2, entrenamiento) de la Ext U-Net Comb V2.1. Lee la arquitectura de la Etapa A (arq_desde_A.py).
# Se lanza con lanzar_comb.sh (dependencia de la Etapa A) o a mano:
#   sbatch --export=ALL,JOB_INDEX=0 sbatch_etapa_B_comb.sh ; sbatch --export=ALL,JOB_INDEX=1 sbatch_etapa_B_comb.sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch_xpu
cd ~/lightning
mkdir -p logs estudios etapa_B_comb
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
unset SUBSET SMOKE DEVICE STUDY_NAME JOURNAL DIR_COMB N_STARTUP TEST_BATCH

export ETAPA=B
export JOB_INDEX=${JOB_INDEX:-0}
if ! [[ "$JOB_INDEX" =~ ^[01]$ ]]; then echo "ERROR: JOB_INDEX tiene que ser 0 o 1 (es '$JOB_INDEX')" >&2; exit 1; fi
export DATASET_DIR=${DATASET_DIR_COMB:-$HOME/data/dataset_Camilo}
export TIMEOUT_S=$((46*3600))
export N_STARTUP=16

echo "job $SLURM_JOB_ID | ETAPA=B | JOB_INDEX=$JOB_INDEX | nodo $(hostname)"

# Arquitectura de la Etapa A (el primer job la escribe; el segundo reutiliza el mismo json)
python arq_desde_A.py || { echo "ERROR: no se pudo fijar la arquitectura desde la Etapa A; se aborta." >&2; exit 1; }

MIN_DEV=${MIN_DEV:-8}
NDEV=$(timeout -k 10 180 python -c "import torch; print(torch.xpu.device_count())" 2>/dev/null | tail -n 1)
echo "device_count: ${NDEV:-<sin respuesta>} (se requieren >= $MIN_DEV)"
if ! [[ "$NDEV" =~ ^[0-9]+$ ]] || [ "$NDEV" -lt "$MIN_DEV" ]; then
    echo "ERROR: dispositivos XPU insuficientes en $(hostname); se aborta." >&2; exit 1
fi

ZE_AFFINITY_MASK=0 TEST_BATCH=32 timeout -k 10 900 python test_comb.py
rc=$?
if [ $rc -ne 0 ]; then
    echo "ERROR: falló test_comb.py (código $rc); se aborta el job." >&2
    ps -eo pid,user,etime,pcpu,pmem,cmd --sort=-pcpu | head -15; exit 1
fi

for i in $(seq 0 $((NDEV-1))); do
    ZE_AFFINITY_MASK=$i OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/NDEV)) \
        python worker_comb.py $i > logs/comb_B_j${JOB_INDEX}_${SLURM_JOB_ID}_w$i.log 2>&1 &
done
wait
