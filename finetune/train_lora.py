"""LoRA fine-tuning (attention layers) on CIDAR or AlpaGasus.

    python train_lora.py --model llama8b --data cidar --r 16 --alpha 32 --out $CKPT_DIR/llama8b_cidar_r16_a32

Saves only the adapter, plus run_config.json with the arguments and training log.
The loss is on the response tokens only.
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from huggingface_hub import try_to_load_from_cache
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

from pairs import DATASETS, load_pairs

MODELS = {  # same short names as inference/models.py
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma2": "google/gemma-2-9b-it",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}


def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tuning")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--data", default="cidar", choices=list(DATASETS))
    p.add_argument("--out", type=Path, required=True, help="adapter folder")
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--train_size", type=int, default=None, help="random subset of N pairs (default: all)")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_length", type=int, default=1024, help="longer examples are truncated")
    p.add_argument("--max_steps", type=int, default=-1, help="stop early (smoke test)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if (args.out / "adapter_model.safetensors").exists():
        raise SystemExit(f"{args.out} already has an adapter")
    set_seed(args.seed)

    path = str(Path(try_to_load_from_cache(MODELS[args.model], "config.json")).parent)
    tok = AutoTokenizer.from_pretrained(path)
    # gemma-2 needs eager attention for its logit soft-capping
    attn = "eager" if args.model == "gemma2" else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation=attn)
    train = Dataset.from_list(load_pairs(args.data, args.train_size, args.seed))
    print(f"{args.model} on {args.data}: {len(train)} pairs, r={args.r} alpha={args.alpha}", flush=True)

    config = SFTConfig(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # effective batch 16
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        bf16=True,
        gradient_checkpointing=True,
        max_length=args.max_length,
        completion_only_loss=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
    )
    lora = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    trainer = SFTTrainer(model=model, args=config, train_dataset=train, processing_class=tok, peft_config=lora)
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(str(args.out))  # adapter only

    peak_gb = round(torch.cuda.max_memory_allocated() / 2**30, 1)
    run = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    run.update(base_model=MODELS[args.model], train_pairs=len(train), peak_gpu_mem_gb=peak_gb,
               log=trainer.state.log_history)
    (args.out / "run_config.json").write_text(json.dumps(run, indent=2))
    print(f"Saved adapter -> {args.out} (peak GPU memory {peak_gb} GB)", flush=True)


if __name__ == "__main__":
    main()
