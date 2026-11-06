"""Collect all *_summary.json into outputs/results.csv and print the main tables.

    python report.py ../outputs
"""
import json
import statistics
import sys
from pathlib import Path

from models import CONDITIONS, MODELS
from results import write_csv

COLS = ["arabicmmlu_0shot", "arabicmmlu_3shot", "mmlu_0shot", "mmlu_3shot"]
ABLATIONS = ["r4_a8", "r8_a16", "r16_a32", "r64_a128", "r16_a16", "r16_a64",
             "r16_a32_n1000", "r16_a32_n2500", "r16_a32_n5000"]

root = Path(sys.argv[1])
rows = []
acc, a_rate = {}, {}
for path in sorted(root.glob("**/*_summary.json")):
    s = json.loads(path.read_text(encoding="utf-8"))
    model = path.stem.removesuffix("_summary")
    run_dir = str(path.parent.parent.relative_to(root))  # baseline or lora/<model>_<run>
    setting = path.parent.name
    a = s["pred_counts"].get("A", 0) / s["questions"]
    rows.append({"run": run_dir, "model": model, "setting": setting, "accuracy": s["accuracy"],
                 "unparsed_rate": s["unparsed_rate"], "pred_A_rate": round(a, 4), "gold_A_rate": s["always_A"]})

    run = run_dir.removeprefix("lora/").removeprefix(model + "_")  # baseline, cidar_r16_a32_s43, ...
    acc[(model, run, setting)] = 100 * s["accuracy"]
    a_rate[(model, run, setting)] = 100 * round(a, 4)
write_csv(root / "results.csv", rows)


def cell(model, runs, col):
    vals = [acc[(model, run, col)] for run in runs if (model, run, col) in acc]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.1f}"
    return f"{statistics.mean(vals):.1f} ± {statistics.stdev(vals):.1f} (n={len(vals)})"


print("## Baseline vs CIDAR LoRA (mean ± std over seeds) vs AlpaGasus (English control)\n")
print("| model | condition | " + " | ".join(COLS) + " |")
print("|" + "---|" * (len(COLS) + 2))
for m in MODELS:
    for name, runs in CONDITIONS.items():
        print(f"| {m} | {name} | " + " | ".join(cell(m, runs, c) for c in COLS) + " |")

print("\n## Ablations (seed 42): ArabicMMLU 0-shot / 3-shot, MMLU 0-shot\n")
print("| setting | " + " | ".join(MODELS) + " |")
print("|" + "---|" * (len(MODELS) + 1))
for a in ["baseline"] + ABLATIONS:
    run = a if a == "baseline" else f"cidar_{a}"
    cells = ["/".join(cell(m, [run], c) for c in COLS[:3]) for m in MODELS]
    print(f"| {a} | " + " | ".join(cells) + " |")

print("\n## How often the model answers 'A' (%), ArabicMMLU 0-shot (gold A = 31.2%)\n")
print("| model | baseline | CIDAR (s42) | AlpaGasus |\n|---|---|---|---|")
for m in MODELS:
    rates = [a_rate.get((m, run, "arabicmmlu_0shot"), float("nan"))
             for run in ["baseline", "cidar_r16_a32", "alpagasus_r16_a32"]]
    print(f"| {m} | " + " | ".join(f"{x:.1f}" for x in rates) + " |")
print(f"\n{len(rows)} rows -> {root / 'results.csv'}")
