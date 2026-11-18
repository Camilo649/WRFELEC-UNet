#!/bin/bash
#SBATCH --job-name=train_f128
#SBATCH --partition=gpunode
#SBATCH --nodes=1
#SBATCH --gres=gpu:intel_xt1550:1
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --output=logs/train_f128_%j.log

source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch_xpu
cd ~/lightning
mkdir -p logs checkpoints
export PYTHONPATH=$PWD

unset SUBSET EPOCAS_TRIAL_A WARMUP_FRAC STUDY_NAME JOURNAL TIMEOUT_S

python -c "import torch; print('Dispositivos XPU disponibles:', torch.xpu.device_count())"

ZE_AFFINITY_MASK=0 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK python train_best_f128.py
