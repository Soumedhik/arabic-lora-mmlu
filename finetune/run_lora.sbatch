#!/bin/bash
#SBATCH --job-name=dllora-train
#SBATCH --partition=ws-ia
#SBATCH -N1
#SBATCH -n12
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --output=../logs/lora_%x_%j.log
# Train one LoRA adapter, then evaluate it on ArabicMMLU + MMLU (0/3-shot).
# Usage (from finetune/): sbatch -J <run> run_lora.sbatch <model> <run> [train_lora.py args]
#   e.g. sbatch -J llama8b_cidar_r16_a32 run_lora.sbatch llama8b llama8b_cidar_r16_a32 --data cidar --r 16 --alpha 32
set -euo pipefail
model=$1; run=$2; shift 2
ckpt=${CKPT_DIR:-/l/users/$USER/checkpoints/dllora}/$run

source /apps/local/anaconda3/conda_init.sh
export HF_HOME=/l/users/$USER/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "host $(hostname)  job $SLURM_JOB_ID  model $model  run $run  args $*"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv

if [[ -f $ckpt/adapter_model.safetensors ]]; then
  echo "=== adapter exists, skipping training"
else
  echo "=== TRAIN START $(date -Is)"
  conda activate /l/users/$USER/envs/mmcqa-train
  python train_lora.py --model "$model" --out "$ckpt" "$@"
  echo "=== TRAIN END $(date -Is)"
fi

echo "=== EVAL START $(date -Is)"
conda activate /l/users/$USER/envs/nlp
cd ../inference
python inference.py --model "$model" --adapter "$ckpt" --output_dir "../outputs/lora/$run"
echo "=== EVAL END $(date -Is)"
