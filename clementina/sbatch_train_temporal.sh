#!/bin/bash
#SBATCH --job-name=train_temporal
#SBATCH --partition=gpunode
#SBATCH --nodes=1
#SBATCH --gres=gpu:intel_xt1550:1
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --output=logs/train_temporal_%j.log

# Uso:  sbatch sbatch_train_temporal.sh --trial 39
#       sbatch sbatch_train_temporal.sh --lr 1e-4 --alpha 0.1 --epocas 100 --batch-size 16
# (los argumentos se pasan tal cual a train_temporal.py)

source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch_xpu
cd ~/lightning
mkdir -p logs checkpoints
export PYTHONPATH=$PWD
export DATASET_DIR=$HOME/data/dataset_Camilo_temporal   # dataset TEMPORAL (9 canales)

unset SUBSET EPOCAS_TRIAL_A WARMUP_FRAC STUDY_NAME JOURNAL TIMEOUT_S

python -c "import torch; print('Dispositivos XPU disponibles:', torch.xpu.device_count())"

ZE_AFFINITY_MASK=0 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK python -u train_temporal.py "$@"
