#!/bin/bash
#SBATCH --job-name=pei5final
#SBATCH --array=0-26
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=logs/final_%A_%a.out
# final models: task = 3 * classifier + part (tuned, default, alt); one CPU thread each, so that the
# training times of all nine classifiers are measured the same way
module load StdEnv/2023 2>/dev/null || true
module load python/3.11 2>/dev/null || module load python
source $HOME/envs/pei5/bin/activate
cd "$SLURM_SUBMIT_DIR"; export PYTHONUNBUFFERED=1
MODELS=(catboost lightgbm xgboost deepmlp ft_transformer vchnet swingnnet fmtunet timesenclip); PARTS=(tuned default alt)
python final.py ${MODELS[$((SLURM_ARRAY_TASK_ID / 3))]} ${PARTS[$((SLURM_ARRAY_TASK_ID % 3))]}
