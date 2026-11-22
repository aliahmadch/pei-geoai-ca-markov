#!/bin/bash
#SBATCH --job-name=pei5select
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=logs/select_%j.out
module load StdEnv/2023 2>/dev/null || true
module load python/3.11 2>/dev/null || module load python
source $HOME/envs/pei5/bin/activate
cd "$SLURM_SUBMIT_DIR"
n=$(ls results/tune/*.json 2>/dev/null | wc -l)
python select_best.py | tee results/selection.txt
if [ "$n" -ne 270 ] && [ "${PEI5_ALLOW_MISSING:-0}" != "1" ]; then echo "only $n of 270 trials finished: resubmit the missing tuning tasks"; exit 1; fi
