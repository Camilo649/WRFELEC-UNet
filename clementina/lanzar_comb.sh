#!/bin/bash
# Lanza la Etapa A y encadena dos jobs de la Etapa B que arrancan recién cuando A terminó (afterany: A cierra por
# tiempo, y un job que termina por TIMEOUT de Slurm no satisface afterok). Si A no dejó suficientes ensayos completados,
# arq_desde_A.py hace fallar los jobs B en el arranque en vez de entrenar con una arquitectura inválida.
#   cd ~/lightning && bash lanzar_comb.sh
cd ~/lightning || exit 1
mkdir -p logs
JA=$(sbatch --parsable sbatch_etapa_A_comb.sh) || exit 1
JB0=$(sbatch --parsable --dependency=afterany:$JA --export=ALL,JOB_INDEX=0 sbatch_etapa_B_comb.sh) || exit 1
JB1=$(sbatch --parsable --dependency=afterany:$JA --export=ALL,JOB_INDEX=1 sbatch_etapa_B_comb.sh) || exit 1
echo "A=$JA | B0=$JB0 (depende de A) | B1=$JB1 (depende de A)"
squeue -u "$USER" -o "%.10i %.14j %.8T %.12M %.20E %R"
