#!/bin/bash
#SBATCH --job-name=optuna_B
#SBATCH --partition=gpunode
#SBATCH --nodes=1
#SBATCH --gres=gpu:intel_xt1550:4
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=logs/optuna_B_%j.log

# Uso:  sbatch --export=ALL,JOB_INDEX=0 sbatch_etapa_B.sh
#       sbatch --export=ALL,JOB_INDEX=1 sbatch_etapa_B.sh     (segundo job, en paralelo)

source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch_xpu
cd ~/lightning
mkdir -p logs estudios etapa_B
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1

# que no se hereden variables de las pruebas
unset SUBSET SMOKE DEVICE STUDY_NAME JOURNAL DIR_B N_STARTUP EPOCAS_TRIAL_A WARMUP_FRAC

export JOB_INDEX=${JOB_INDEX:-0}
export TIMEOUT_S=$((46*3600))      # no lanza trials nuevos pasadas 46 h

echo "job $SLURM_JOB_ID | JOB_INDEX=$JOB_INDEX | nodo $(hostname)"
python -c "import torch; print('device_count:', torch.xpu.device_count())"

NDEV=8
for i in $(seq 0 $((NDEV-1))); do
    ZE_AFFINITY_MASK=$i OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/NDEV)) \
        python worker_B.py $i > logs/B_j${JOB_INDEX}_w$i.log 2>&1 &
done
wait
