#!/bin/bash
#SBATCH --job-name=pei5tune
#SBATCH --array=0-269
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=logs/tune_%A_%a.out
# one hyperparameter trial of one classifier per task: task = 30 * classifier + trial (CPU only; VCHNet and FMTUNet
# trials take up to about 1 h, the other classifiers a few minutes)
module load StdEnv/2023 2>/dev/null || true
module load python/3.11 2>/dev/null || module load python
source $HOME/envs/pei5/bin/activate
cd "$SLURM_SUBMIT_DIR"; export PYTHONUNBUFFERED=1
MODELS=(catboost lightgbm xgboost deepmlp ft_transformer vchnet swingnnet fmtunet timesenclip)
python tune.py ${MODELS[$((SLURM_ARRAY_TASK_ID / 30))]} $((SLURM_ARRAY_TASK_ID % 30)) --threads ${SLURM_CPUS_PER_TASK:-1}
