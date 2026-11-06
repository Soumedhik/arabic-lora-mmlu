"""Where does CIDAR LoRA lose ArabicMMLU accuracy? Uses the saved per-question outputs (no GPU).

    python error_analysis.py ../outputs --shots 0     # writes <outputs>/error_analysis_0shot.csv

Accuracy is split three ways: by subject group, by task, and by whether the gold answer is 'A'.
CIDAR is the mean over its 3 seeds.
"""
import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from models import CONDITIONS, MODELS
from results import read_jsonl, unpack_outputs, write_csv

p = argparse.ArgumentParser()
p.add_argument("outputs", type=Path)
p.add_argument("--shots", type=int, default=0)
args = p.parse_args()
unpack_outputs()


def gold_is_a(r):
    return "gold=A" if r["gold"] == "A" else "gold!=A"


ANALYSES = [
    ("group", "arabicmmlu", lambda r: r["group"]),
    ("task", "arabicmmlu", lambda r: r["task"]),
    ("gold_is_A", "arabicmmlu", gold_is_a),
    ("gold_is_A", "mmlu", gold_is_a),
]


def accuracy(model, run, bench, key):
    """Accuracy (%) per group for one run."""
    folder = args.outputs / "baseline" if run == "baseline" else args.outputs / "lora" / f"{model}_{run}"
    groups = defaultdict(list)
    for r in read_jsonl(folder / f"{bench}_{args.shots}shot" / f"{model}.jsonl"):
        groups[key(r)].append(r["correct"])
    return {g: 100 * sum(v) / len(v) for g, v in groups.items()}


rows = []
for m in MODELS:
    for cond, runs in CONDITIONS.items():
        for name, bench, key in ANALYSES:
            per_run = [accuracy(m, run, bench, key) for run in runs]
            for g in sorted(per_run[0]):
                mean = statistics.mean(a[g] for a in per_run)
                rows.append({"model": m, "condition": cond, "analysis": name, "benchmark": bench,
                             "group": g, "accuracy": round(mean, 2)})

out = args.outputs / f"error_analysis_{args.shots}shot.csv"
write_csv(out, rows)

# Short printout: subject groups, baseline -> CIDAR
acc = {(r["model"], r["condition"], r["analysis"], r["benchmark"], r["group"]): r["accuracy"] for r in rows}
groups = sorted({r["group"] for r in rows if r["analysis"] == "group"})
print(f"## ArabicMMLU {args.shots}-shot by subject group: baseline -> CIDAR\n")
print("| model | " + " | ".join(groups) + " |")
print("|" + "---|" * (len(groups) + 1))
for m in MODELS:
    cells = []
    for g in groups:
        b, c = acc[(m, "baseline", "group", "arabicmmlu", g)], acc[(m, "CIDAR", "group", "arabicmmlu", g)]
        cells.append(f"{b:.1f} → {c:.1f}")
    print(f"| {m} | " + " | ".join(cells) + " |")
print(f"\n-> {out}")
