#!/bin/bash
#SBATCH --job-name=dllora-eval
#SBATCH --partition=ws-ia
#SBATCH -N1
#SBATCH -n12
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=../logs/eval_%j.log
# Usage (from inference/): sbatch run_eval.sbatch <model> [<model> ...] [-- extra inference.py args]
set -uo pipefail
models=(); extra=()
while (($#)); do [[ $1 == -- ]] && { shift; extra=("$@"); break; }; models+=("$1"); shift; done

source /apps/local/anaconda3/conda_init.sh
conda activate /l/users/$USER/envs/nlp
export HF_HOME=/l/users/$USER/hf_cache HF_HUB_OFFLINE=1
echo "host $(hostname)  job $SLURM_JOB_ID  models ${models[*]}  extra ${extra[*]}"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv

status=0
for m in "${models[@]}"; do
  echo "=== START $m $(date -Is)"
  python inference.py --model "$m" "${extra[@]}"
  rc=$?; echo "=== END $m rc=$rc $(date -Is)"; ((rc)) && status=1
done
exit $status
