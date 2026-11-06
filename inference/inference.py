"""Greedy multiple-choice evaluation on ArabicMMLU and English MMLU.

    python inference.py --model llama8b
    python inference.py --model qwen7b --benchmarks mmlu --shots 0 --limit 50
    python inference.py --model llama8b --adapter <adapter dir> --output_dir ../outputs/lora/<run>

Writes <output_dir>/<benchmark>_<k>shot/<model>.jsonl and <model>_summary.json.
Settings that already have output are skipped.
"""
import argparse
import os
from collections import defaultdict
from pathlib import Path

from data import LOADERS
from models import MODELS, load_llm
from prompts import encode
from results import make_records, unpack_outputs, write_results

OUT_DIR = Path(os.environ.get("OUT_DIR", Path(__file__).resolve().parent.parent / "outputs" / "baseline"))


def parse_args():
    p = argparse.ArgumentParser(description="ArabicMMLU / MMLU multiple-choice evaluation")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--benchmarks", nargs="+", choices=list(LOADERS), default=list(LOADERS))
    p.add_argument("--shots", nargs="+", type=int, default=[0, 3], help="few-shot examples from the dev split")
    p.add_argument("--output_dir", type=Path, default=OUT_DIR)
    p.add_argument("--adapter", type=Path, default=None, help="LoRA adapter folder from finetune/train_lora.py")
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="only the first N questions")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def dev_examples(bench: str, k: int) -> dict:
    """Dev questions grouped by task (the few-shot pool)."""
    dev = defaultdict(list)
    for r in LOADERS[bench]("dev"):
        dev[r["task"]].append(r)
    if k > min(len(v) for v in dev.values()):
        raise SystemExit(f"{bench}: some task has fewer than {k} dev examples")
    return dev


def run(args, llm, bench: str, k: int) -> None:
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    out_path = args.output_dir / f"{bench}_{k}shot" / f"{args.model}.jsonl"
    if out_path.exists():
        print(f"skip {out_path}", flush=True)
        return

    test = LOADERS[bench]("test")[: args.limit]
    dev = dev_examples(bench, k)
    tok = llm.get_tokenizer()
    prompts = [{"prompt_token_ids": encode(tok, bench, r, dev[r["task"]][:k])} for r in test]
    print(f"{bench} {k}-shot: {len(test)} questions", flush=True)

    params = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)
    lora = LoRARequest(args.adapter.name, 1, str(args.adapter)) if args.adapter else None
    outputs = llm.generate(prompts, params, lora_request=lora)

    adapter = str(args.adapter) if args.adapter else None
    meta = {"model": MODELS[args.model], "adapter": adapter, "benchmark": bench, "shots": k}
    write_results(out_path, make_records(test, outputs), meta)


def main():
    args = parse_args()
    unpack_outputs()  # so finished runs are skipped
    llm = load_llm(args.model, args.seed, lora=args.adapter is not None)
    for bench in args.benchmarks:
        for k in args.shots:
            run(args, llm, bench, k)


if __name__ == "__main__":
    main()
