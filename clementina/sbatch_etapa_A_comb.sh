#!/bin/bash
#SBATCH --job-name=optuna_A_comb
#SBATCH --partition=gpunode
#SBATCH --nodes=1
#SBATCH --gres=gpu:intel_xt1550:4
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --exclusive
#SBATCH --exclude=cn045,cn072,cn021,cn033
#SBATCH --output=logs/optuna_A_comb_%j.log

# Etapa A (Fase 1, arquitectura) de la Ext Clip U-Net Comb V2.1, con el protocolo de la Fase 1 de la Comb V2 (ajustado, ver objetivo_comb.py): 46 h, 8 tiles.
# Uso: sbatch sbatch_etapa_A_comb.sh      (o todo junto con lanzar_comb.sh)

source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch_xpu
cd ~/lightning
mkdir -p logs estudios etapa_A_comb
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
unset SUBSET SMOKE DEVICE STUDY_NAME JOURNAL DIR_COMB N_STARTUP TEST_BATCH

export ETAPA=A JOB_INDEX=0
export DATASET_DIR=${DATASET_DIR_COMB:-$HOME/data/dataset_Camilo}   # Ext Clip: 3 canales, LIGHTDENS <= 150
export TIMEOUT_S=$((46*3600))                       # 46 h: mismo presupuesto que la Fase 1 de la Ext Clip FACL
export N_STARTUP=10

echo "job $SLURM_JOB_ID | ETAPA=A | nodo $(hostname) | DATASET_DIR=$DATASET_DIR"

MIN_DEV=${MIN_DEV:-8}
NDEV=$(timeout -k 10 180 python -c "import torch; print(torch.xpu.device_count())" 2>/dev/null | tail -n 1)
echo "device_count: ${NDEV:-<sin respuesta>} (se requieren >= $MIN_DEV)"
if ! [[ "$NDEV" =~ ^[0-9]+$ ]] || [ "$NDEV" -lt "$MIN_DEV" ]; then
    echo "ERROR: dispositivos XPU insuficientes en $(hostname); se aborta." >&2; exit 1
fi

# Chequeo de infraestructura con una arquitectura intermedia (las que no entren en memoria se podan solas como OOM)
ZE_AFFINITY_MASK=0 TEST_ARQ=med timeout -k 10 1200 python test_comb.py
rc=$?
if [ $rc -ne 0 ]; then
    echo "ERROR: falló test_comb.py (código $rc); se aborta el job." >&2
    ps -eo pid,user,etime,pcpu,pmem,cmd --sort=-pcpu | head -15; exit 1
fi

for i in $(seq 0 $((NDEV-1))); do
    ZE_AFFINITY_MASK=$i OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/NDEV)) \
        python worker_comb.py $i > logs/comb_A_${SLURM_JOB_ID}_w$i.log 2>&1 &
done
wait
