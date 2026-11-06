"""Position-debiased (circular) 0-shot evaluation.

Each question is asked once per cyclic rotation of its options, so the gold answer appears at
every letter exactly once and answer-position bias cannot change the score.

    python debiased.py --model llama8b                  # baseline + 3 CIDAR seeds + AlpaGasus
    python debiased.py --model qwen7b --conditions baseline --limit 50

Adapters are read from $CKPT_DIR/<model>_<condition>. Output, one file per rotation j
(rotation 0 is the original order), summarised by debiased_report.py:
    <output_dir>/<condition>/<benchmark>_rot<j>/<model>.jsonl
"""
import argparse
import os
from pathlib import Path

from data import LOADERS
from models import CONDITIONS, MODELS, load_llm
from prompts import LETTERS, encode
from results import make_records, unpack_outputs, write_results

CKPT_DIR = Path(os.environ.get("CKPT_DIR", f"/l/users/{os.environ.get('USER')}/checkpoints/dllora"))
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "debiased"
ALL_CONDITIONS = [run for runs in CONDITIONS.values() for run in runs]


def rotate(row: dict, j: int) -> dict:
    """Shift the options left by j (option j becomes A) and move the gold letter with its option."""
    gold = (LETTERS.index(row["gold"]) - j) % len(row["options"])
    return {**row, "options": row["options"][j:] + row["options"][:j], "gold": LETTERS[gold]}


def parse_args():
    p = argparse.ArgumentParser(description="Circular (option-rotated) 0-shot evaluation")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS, help="'baseline' or an adapter run name")
    p.add_argument("--benchmarks", nargs="+", choices=list(LOADERS), default=list(LOADERS))
    p.add_argument("--output_dir", type=Path, default=OUT_DIR)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    unpack_outputs()  # so finished runs are skipped
    adapters = {c: CKPT_DIR / f"{args.model}_{c}" for c in args.conditions if c != "baseline"}
    for c, path in adapters.items():
        if not (path / "adapter_model.safetensors").exists():
            raise SystemExit(f"missing adapter for {c}: {path}")

    llm = load_llm(args.model, args.seed, lora=True)  # one engine for all conditions
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    tok = llm.get_tokenizer()
    params = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)

    for bench in args.benchmarks:
        test = LOADERS[bench]("test")[: args.limit]
        for j in range(len(LETTERS)):
            rows = [rotate(r, j) for r in test if j < len(r["options"])]
            if not rows:  # e.g. rotation 4 when no question has 5 options
                continue
            prompts = [{"prompt_token_ids": encode(tok, bench, r, [])} for r in rows]
            for i, c in enumerate(args.conditions):
                out_path = args.output_dir / c / f"{bench}_rot{j}" / f"{args.model}.jsonl"
                if out_path.exists():
                    print(f"skip {out_path}", flush=True)
                    continue
                print(f"{c} {bench} rotation {j}: {len(rows)} questions", flush=True)
                lora = LoRARequest(c, i + 1, str(adapters[c])) if c in adapters else None
                outputs = llm.generate(prompts, params, lora_request=lora, use_tqdm=False)
                adapter = str(adapters[c]) if c in adapters else None
                meta = {"model": MODELS[args.model], "adapter": adapter, "benchmark": bench, "shots": 0, "rotation": j}
                write_results(out_path, make_records(rows, outputs), meta)


if __name__ == "__main__":
    main()
