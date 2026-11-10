#!/bin/bash
#SBATCH --job-name=exams
#SBATCH --partition=cscc-gpu-p
#SBATCH -N1
#SBATCH -n12
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=../logs/exams_%x_%j.log
# Held-out Arabic EXAMS questions, log-probability scoring, 0-shot and circular.
# Usage (from inference/): sbatch run_exams.sbatch <model>:<run> [<model>:<run> ...]
#   <run> = base for the model without an adapter, or the name of a trained run, e.g.
#   sbatch run_exams.sbatch llama8b:base llama8b:llama8b_mcqar_r16_a32 qwen7b:base
set -euo pipefail
source /apps/local/anaconda3/conda_init.sh
conda activate /l/users/$USER/envs/nlp
export HF_HOME=/l/users/$USER/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
ckpts=${CKPT_DIR:-/l/users/$USER/checkpoints/dllora}
for spec in "$@"; do
  model=${spec%%:*}; run=${spec#*:}
  echo "=== $model $run $(date -Is)"
  if [[ $run == base ]]; then
    args=()
  else
    args=(--adapter "$ckpts/$run" --output_dir "../outputs/logprob/lora/$run")
  fi
  python logprob.py --model "$model" --benchmarks exams_ar --shots 0 "${args[@]}"
  python logprob.py --model "$model" --benchmarks exams_ar --circular "${args[@]}"
done
echo "=== DONE $(date -Is)"
