#!/bin/bash
#SBATCH --job-name=optuna_test
#SBATCH --partition=testing
#SBATCH --nodelist=cn053
#SBATCH --gres=gpu:intel_xt1550:4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:20:00
#SBATCH --output=test_%j.log

source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch_xpu
cd ~/lightning
export PYTHONPATH=$PWD
export STUDY_NAME=TEST JOURNAL=estudios/optuna_TEST.log
export EPOCAS_TRIAL_A=2 WARMUP_FRAC=0 TIMEOUT_S=480   # ~8 min

python test_xpu_facl.py || exit 1

NDEV=8
for i in $(seq 0 $((NDEV-1))); do
    ZE_AFFINITY_MASK=$i OMP_NUM_THREADS=$((64/NDEV)) \
        python worker_optuna.py $i > worker_$i.log 2>&1 &
done
sleep 120
xpu-smi dump -d -1 -m 1,5 -n 3 > xpu_dump.log 2>&1
wait
