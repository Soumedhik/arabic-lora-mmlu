# LoRA fine-tuning on Arabic instructions: ArabicMMLU vs MMLU

Five instruction-tuned models (Llama-3.1-8B, Mistral-7B-v0.3, Gemma-2-9B, Qwen2.5-7B, Qwen2.5-1.5B)
are fine-tuned with LoRA on CIDAR (Arabic) and on AlpaGasus (its English source, as a control), then
evaluated on ArabicMMLU and English MMLU, 0- and 3-shot, with greedy decoding.

A second line of experiments scores answers by the log-probability of the option letter (as in the
ArabicMMLU paper) and adds a third training set, `mcq_ar`: 14.6k Arabic multiple-choice questions
(the Arabic translation of MMLU) in the ArabicMMLU prompt format with shuffled options and a
one-letter answer. Held-out Arabic EXAMS questions are used as an extra test set. The English MMLU
score of `mcq_ar` runs is contaminated (same questions), so only ArabicMMLU and EXAMS count.

## Layout

```
finetune/
  train_lora.py         LoRA training (attention layers, loss on responses only)
  pairs.py              CIDAR / AlpaGasus / mcq_ar loading
  run_lora.sbatch       train one adapter, then evaluate it (vLLM, generation)
  run_lora_logprob.sbatch   train one adapter, then log-probability evaluation incl. circular and EXAMS
inference/
  inference.py          main evaluation (0/3-shot, optional --adapter)
  debiased.py           circular evaluation: every rotation of the options
  logprob.py            log-probability evaluation (0/3-shot, --circular, --adapter, merged weights)
  run_logprob.sbatch    baselines with logprob.py;  run_exams.sbatch: held-out Arabic EXAMS
  compare_runs.py       base vs fine-tuned runs with paired tests (no GPU)
  subject_analysis.py   per-task / level / consensus / position-bias tables (no GPU)
  data.py prompts.py    benchmark loading and prompt format (exams_ar only with --benchmarks)
  models.py             model names, run groups, vLLM loading
  results.py            answer parsing and result files
  report.py             results.csv + main tables
  debiased_report.py error_analysis.py fertility.py   analyses
outputs/
  baseline/  lora/<model>_<run>/  debiased/<run>/   _summary.json per run
  logprob/baseline/  logprob/lora/<run>/            same for the log-probability evaluation
  raw_outputs.zip       per-question .jsonl answers, unzipped automatically on run
  results.csv  debiased_results.csv  error_analysis_{0,3}shot.csv  fertility.csv
```

Adapter runs are named `<model>_<data>_r<rank>_a<alpha>[_n<train size>][_s<seed>]`.
Runs without `_s` use seed 42. The `mcq_ar` runs are called `<model>_mcqar_r16_a32`; learning-rate
variants add `_lr<value>`. `mcq_ar` needs `alexandrainst/m_mmlu` (Arabic files) in the cache, and
the EXAMS test needs `OALL/Arabic_EXAMS`.

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

# log-probability evaluation (one environment: torch, transformers, peft, trl)
sbatch -J llama8b_mcqar_r16_a32 ../finetune/run_lora_logprob.sbatch llama8b llama8b_mcqar_r16_a32 --data mcq_ar --r 16 --alpha 32
sbatch run_exams.sbatch llama8b:base qwen7b:base
python compare_runs.py ../outputs --model llama8b --runs llama8b_cidar_r16_a32 llama8b_mcqar_r16_a32
python subject_analysis.py ../outputs --shots 0

python report.py ../outputs
python debiased_report.py ../outputs/debiased
python error_analysis.py ../outputs --shots 0
python fertility.py
```

The analysis scripts only read the saved outputs and do not need a GPU. The per-question `.jsonl`
files are stored in `outputs/raw_outputs.zip`. `inference.py`, `debiased.py`, `debiased_report.py`
and `error_analysis.py` unzip any missing ones when they start.
