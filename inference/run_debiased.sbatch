#!/bin/bash
#SBATCH --job-name=dllora-debiased
#SBATCH --partition=ws-ia
#SBATCH -N1
#SBATCH -n12
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --output=../logs/debiased_%x_%j.log
# Circular 0-shot evaluation of one model (baseline + adapters). Usage (from inference/):
#   sbatch -J debiased_<model> run_debiased.sbatch <model> [extra debiased.py args]
set -euo pipefail
model=$1; shift

source /apps/local/anaconda3/conda_init.sh
conda activate /l/users/$USER/envs/nlp
export HF_HOME=/l/users/$USER/hf_cache HF_HUB_OFFLINE=1
echo "host $(hostname)  job $SLURM_JOB_ID  model $model  args $*"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
echo "=== START $(date -Is)"
python debiased.py --model "$model" "$@"
echo "=== END $(date -Is)"
