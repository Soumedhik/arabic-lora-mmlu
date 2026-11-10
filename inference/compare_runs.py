"""Compare fine-tuned runs with the base model on log-probability results (no GPU needed).

    python compare_runs.py ../outputs --model llama8b --runs llama8b_cidar_r16_a32 llama8b_mcqar_r16_a32
    python compare_runs.py ../outputs --model mistral --runs mistral_mcqar_r16_a32_lr1e-4 --benchmarks arabicmmlu exams_ar

Reads <outputs>/logprob/baseline and <outputs>/logprob/lora/<run>. For each benchmark it prints the
0-shot and 3-shot accuracy, the circular accuracy (mean over all rotations of the options), the share of
answers that are 'A', and for ArabicMMLU the number of tasks that improved. Differences come with a
paired McNemar z (0-shot, 3-shot) and a 95% interval (circular), computed over the same questions.
"""
import argparse
import math
import statistics
from collections import defaultdict
from pathlib import Path

from results import read_jsonl


def accuracy(recs):
    return 100 * statistics.mean(r["correct"] for r in recs)


def share_a(recs):
    return 100 * statistics.mean(r["pred"] == "A" for r in recs)


def mcnemar_z(base, run):
    gained = sum(not b["correct"] and r["correct"] for b, r in zip(base, run))
    lost = sum(b["correct"] and not r["correct"] for b, r in zip(base, run))
    return (gained - lost) / math.sqrt(gained + lost) if gained + lost else 0.0


def circular(folder: Path, bench: str, model: str):
    """Per question: mean correctness over its rotations, and the question's task."""
    per_q, task = defaultdict(list), {}
    for f in sorted(folder.glob(f"{bench}_rot*/{model}.jsonl")):
        for r in read_jsonl(f):
            per_q[r["id"]].append(r["correct"])
            task[r["id"]] = r["task"]
    return {i: statistics.mean(v) for i, v in per_q.items()}, task


def main():
    p = argparse.ArgumentParser()
    p.add_argument("outputs", type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--runs", nargs="+", required=True, help="adapter run names under logprob/lora/")
    p.add_argument("--benchmarks", nargs="+", default=["arabicmmlu", "exams_ar", "mmlu"])
    args = p.parse_args()
    root = args.outputs / "logprob"
    base_dir = root / "baseline"

    for bench in args.benchmarks:
        base_circ, task = circular(base_dir, bench, args.model)
        if not base_circ:
            print(f"\n{bench}: no baseline results, skipped")
            continue
        base = {k: read_jsonl(base_dir / f"{bench}_{k}shot" / f"{args.model}.jsonl")
                for k in (0, 3) if (base_dir / f"{bench}_{k}shot" / f"{args.model}.jsonl").exists()}
        print(f"\n{bench} / {args.model}: base 0-shot {accuracy(base[0]):.1f}"
              + (f", 3-shot {accuracy(base[3]):.1f}" if 3 in base else "")
              + f", circular {100 * statistics.mean(base_circ.values()):.1f}, answers A {share_a(base[0]):.0f}%")
        for run in args.runs:
            folder = root / "lora" / run
            run_circ, _ = circular(folder, bench, args.model)
            if len(run_circ) != len(base_circ):
                print(f"  {run}: results incomplete, skipped")
                continue
            parts = []
            for k in sorted(base):
                path = folder / f"{bench}_{k}shot" / f"{args.model}.jsonl"
                if path.exists():
                    recs = read_jsonl(path)
                    parts.append(f"{k}-shot {accuracy(recs):.1f} ({accuracy(recs) - accuracy(base[k]):+.1f}, "
                                 f"z={mcnemar_z(base[k], recs):+.1f})")
            diffs = [run_circ[i] - base_circ[i] for i in base_circ]
            half = 1.96 * statistics.pstdev(diffs) / math.sqrt(len(diffs))
            line = (f"  {run}: " + ", ".join(parts)
                    + f", circular {100 * statistics.mean(run_circ.values()):.1f} "
                      f"({100 * statistics.mean(diffs):+.1f} ± {100 * half:.1f})"
                    + f", answers A {share_a(read_jsonl(folder / f'{bench}_0shot' / f'{args.model}.jsonl')):.0f}%")
            if bench == "arabicmmlu":
                by_task = defaultdict(list)
                for i in base_circ:
                    by_task[task[i]].append(run_circ[i] - base_circ[i])
                deltas = [statistics.mean(v) for v in by_task.values() if len(v) >= 100]
                line += f", tasks improved {sum(d > 0 for d in deltas)}/{len(deltas)}"
            print(line)


if __name__ == "__main__":
    main()
