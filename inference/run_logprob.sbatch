#!/bin/bash
#SBATCH --job-name=dllora-logprob
#SBATCH --partition=cscc-gpu-p
#SBATCH -N1
#SBATCH -n12
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=../logs/logprob_%j.log
# Log-probability evaluation of baseline models. Usage (from inference/):
#   sbatch run_logprob.sbatch <model> [<model> ...] [-- extra logprob.py args]
#   e.g. sbatch run_logprob.sbatch llama8b mistral gemma2 qwen7b qwen1.5b
#        sbatch run_logprob.sbatch llama8b -- --circular
set -uo pipefail
models=(); extra=()
while (($#)); do [[ $1 == -- ]] && { shift; extra=("$@"); break; }; models+=("$1"); shift; done

source /apps/local/anaconda3/conda_init.sh
conda activate /l/users/$USER/envs/nlp
export HF_HOME=/l/users/$USER/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "host $(hostname)  job $SLURM_JOB_ID  models ${models[*]}  extra ${extra[*]}"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv

status=0
for m in "${models[@]}"; do
  echo "=== START $m $(date -Is)"
  python logprob.py --model "$m" "${extra[@]}"
  rc=$?; echo "=== END $m rc=$rc $(date -Is)"; ((rc)) && status=1
done
exit $status
