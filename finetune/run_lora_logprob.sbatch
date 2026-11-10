#!/bin/bash
#SBATCH --job-name=dllora-train
#SBATCH --partition=cscc-gpu-p
#SBATCH -N1
#SBATCH -n12
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --output=../logs/lora_%x_%j.log
# Train one LoRA adapter, then score it with the log-probability evaluation (logprob.py):
# 0/3-shot on ArabicMMLU + MMLU, then the circular (all option rotations) 0-shot runs.
# Usage (from finetune/): sbatch -J <run> run_lora_logprob.sbatch <model> <run> [train_lora.py args]
#   e.g. sbatch -J qwen7b_cidar_r16_a32 run_lora_logprob.sbatch qwen7b qwen7b_cidar_r16_a32 --data cidar --r 16 --alpha 32
# One conda env (torch, transformers, peft, trl) is used for training and evaluation.
set -euo pipefail
model=$1; run=$2; shift 2
ckpt=${CKPT_DIR:-/l/users/$USER/checkpoints/dllora}/$run

source /apps/local/anaconda3/conda_init.sh
conda activate /l/users/$USER/envs/nlp
export HF_HOME=/l/users/$USER/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "host $(hostname)  job $SLURM_JOB_ID  model $model  run $run  args $*"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv

if [[ -f $ckpt/adapter_model.safetensors ]]; then
  echo "=== adapter exists, skipping training"
else
  echo "=== TRAIN START $(date -Is)"
  python train_lora.py --model "$model" --out "$ckpt" "$@"
  echo "=== TRAIN END $(date -Is)"
fi

echo "=== EVAL START $(date -Is)"
cd ../inference
out=../outputs/logprob/lora/$run
python logprob.py --model "$model" --adapter "$ckpt" --output_dir "$out"
python logprob.py --model "$model" --adapter "$ckpt" --output_dir "$out" --circular
python logprob.py --model "$model" --adapter "$ckpt" --output_dir "$out" --benchmarks exams_ar --shots 0
python logprob.py --model "$model" --adapter "$ckpt" --output_dir "$out" --benchmarks exams_ar --circular
echo "=== EVAL END $(date -Is)"
