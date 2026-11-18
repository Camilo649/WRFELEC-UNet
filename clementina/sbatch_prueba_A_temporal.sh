#!/bin/bash
#SBATCH --job-name=prueba_A_temporal
#SBATCH --partition=testing
#SBATCH --nodelist=cn053
#SBATCH --gres=gpu:intel_xt1550:4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:20:00
#SBATCH --output=logs/prueba_A_temporal_%j.log

# Prueba de la Etapa A TEMPORAL en la partición testing (límite 20 min): igual que sbatch_test.sh
# pero con el dataset de 9 canales y worker_A_temporal.py. 8 workers, trials de 2 épocas sobre un
# subconjunto chico, estudio y journal APARTE (no toca el de la corrida real).
# Uso:  sbatch sbatch_prueba_A_temporal.sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch_xpu
cd ~/lightning
mkdir -p logs estudios
export PYTHONPATH=$PWD

export DATASET_DIR=$HOME/data/dataset_Camilo_temporal   # dataset TEMPORAL (9 canales)
export SUBSET=512                                       # 512 muestras de train y 128 de val: trials de segundos, así entran varios por worker
export STUDY_NAME=TEST_A_temporal JOURNAL=estudios/TEST_A_temporal.log
export EPOCAS_TRIAL_A=2 WARMUP_FRAC=0 TIMEOUT_S=480     # ~8 min

# Chequeo: el dataset tiene que tener 9 canales
python - <<'PYEOF' || exit 1
import os, numpy as np
d = os.environ["DATASET_DIR"]
n = int(np.load(os.path.join(d, "stats.npz"))["min_X"].shape[0])
print(f"DATASET_DIR={d} | canales={n}")
assert n == 9, f"se esperaban 9 canales y hay {n}"
PYEOF

python test_xpu_facl.py || exit 1

NDEV=8
for i in $(seq 0 $((NDEV-1))); do
    ZE_AFFINITY_MASK=$i OMP_NUM_THREADS=$((64/NDEV)) \
        python worker_A_temporal.py $i > logs/prueba_A_temporal_${SLURM_JOB_ID}_w$i.log 2>&1 &
done
sleep 120
xpu-smi dump -d -1 -m 1,5 -n 3 > logs/prueba_A_temporal_${SLURM_JOB_ID}_xpu_dump.log 2>&1
wait
