# LoRA fine-tuning on Arabic instructions: ArabicMMLU vs MMLU

Five instruction-tuned models (Llama-3.1-8B, Mistral-7B-v0.3, Gemma-2-9B, Qwen2.5-7B, Qwen2.5-1.5B)
are fine-tuned with LoRA on CIDAR (Arabic) and on AlpaGasus (its English source, as a control), then
evaluated on ArabicMMLU and English MMLU, 0- and 3-shot, with greedy decoding.

## Layout

```
finetune/
  train_lora.py         LoRA training (attention layers, loss on responses only)
  pairs.py              CIDAR / AlpaGasus loading
  run_lora.sbatch       train one adapter, then evaluate it
inference/
  inference.py          main evaluation (0/3-shot, optional --adapter)
  debiased.py           circular evaluation: every rotation of the options
  data.py prompts.py    benchmark loading and prompt format
  models.py             model names, run groups, vLLM loading
  results.py            answer parsing and result files
  report.py             results.csv + main tables
  debiased_report.py error_analysis.py fertility.py   analyses
outputs/
  baseline/  lora/<model>_<run>/  debiased/<run>/   _summary.json per run
  raw_outputs.zip       per-question .jsonl answers, unzipped automatically on run
  results.csv  debiased_results.csv  error_analysis_{0,3}shot.csv  fertility.csv
```

Adapter runs are named `<model>_<data>_r<rank>_a<alpha>[_n<train size>][_s<seed>]`.
Runs without `_s` use seed 42.

## Setup

Two conda environments (versions in `requirements.txt`): one for evaluation (vLLM) and one for training (TRL/PEFT).
Models and datasets are read from the local Hugging Face cache only, so set `HF_HOME` and download them first.

| Variable | Default | Used for |
|---|---|---|
| `HF_HOME` | – | Hugging Face cache |
| `CKPT_DIR` | `/l/users/$USER/checkpoints/dllora` | LoRA adapters |
| `OUT_DIR` | `outputs/baseline` (`outputs/` for fertility) | evaluation output |
| `GPU_MEM_UTIL` | `0.80` | vLLM GPU memory share |

## Running

From `finetune/`:

```
sbatch -J llama8b_cidar_r16_a32 run_lora.sbatch llama8b llama8b_cidar_r16_a32 --data cidar --r 16 --alpha 32
```

From `inference/`:

```
sbatch run_eval.sbatch llama8b mistral gemma2 qwen7b qwen1.5b      # baselines
sbatch -J debiased_llama8b run_debiased.sbatch llama8b             # circular evaluation

python report.py ../outputs
python debiased_report.py ../outputs/debiased
python error_analysis.py ../outputs --shots 0
python fertility.py
```

The analysis scripts only read the saved outputs and do not need a GPU. The per-question `.jsonl`
files are stored in `outputs/raw_outputs.zip`. `inference.py`, `debiased.py`, `debiased_report.py`
and `error_analysis.py` unzip any missing ones when they start.
