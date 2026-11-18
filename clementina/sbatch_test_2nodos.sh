#!/bin/bash
#SBATCH --job-name=test2n
#SBATCH --partition=testing
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --output=logs/test2n_%j.log

source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch_xpu
cd ~/lightning
mkdir -p logs estudios
export PYTHONPATH=$PWD
export JOURNAL=estudios/optuna_TEST2N.log STUDY_NAME=TEST2N
rm -f "$JOURNAL" "$JOURNAL.lock"

# 8 workers por nodo (ids globales 0-15). Variables: JOB_INDEX, TIMEOUT_S, ZOMBIES
fase() {
  export JOB_INDEX=$1 TIMEOUT_S=$2 ZOMBIES=$3
  srun --ntasks=2 --ntasks-per-node=1 --cpus-per-task=16 bash -c '
    for i in $(seq 0 7); do
      id=$((SLURM_NODEID*8+i))
      python test_storage_2nodos.py $id > logs/t2n_j${JOB_INDEX}_w${id}.log 2>&1 &
    done
    wait'
}

echo "== Fase 1: 16 workers en 2 nodos, con 2 workers que mueren a mitad de trial =="
fase 0 240 1
python check_estudio.py

echo "== Limpieza: RUNNING -> FAIL y re-encolar =="
python check_estudio.py --limpiar

echo "== Fase 2: un 'segundo job' continúa el mismo estudio =="
fase 1 90 0
python check_estudio.py
