"""Summarise debiased.py outputs: original-order vs circular accuracy, and how often 'A' is answered.

    python debiased_report.py ../outputs/debiased    # writes <dir>/debiased_results.csv, prints tables

original = accuracy with the options in their original order (rotation 0)
circular = accuracy averaged over all rotations of each question, so position bias cancels
all_rot  = share of questions answered correctly in every rotation
pred_A   = share of answers that are 'A' (unbiased: about 26.7% Arabic, 25% English)
"""
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from models import CONDITIONS, MODELS
from results import read_jsonl, unpack_outputs, write_csv

BENCHES = ["arabicmmlu", "mmlu"]


def summarise(cond_dir: Path, model: str, bench: str) -> dict | None:
    files = sorted(cond_dir.glob(f"{bench}_rot*/{model}.jsonl"))
    if not files:
        return None
    per_q, preds, original = defaultdict(list), [], None
    for f in files:
        recs = read_jsonl(f)
        for r in recs:
            per_q[r["id"]].append(r)
            preds.append(r["pred"])
        if f.parent.name.endswith("_rot0"):
            original = sum(r["correct"] for r in recs) / len(recs)

    complete = [rs for rs in per_q.values() if len(rs) == rs[0]["n_options"]]
    if len(complete) < len(per_q):
        print(f"warning: {cond_dir.name}/{model}/{bench}: {len(per_q) - len(complete)} questions lack rotations")
    return {
        "model": model, "condition": cond_dir.name, "benchmark": bench, "questions": len(complete),
        "original": round(original, 4) if original is not None else None,
        "circular": round(statistics.mean(sum(r["correct"] for r in rs) / len(rs) for rs in complete), 4),
        "all_rot": round(statistics.mean(all(r["correct"] for r in rs) for rs in complete), 4),
        "pred_A": round(sum(p == "A" for p in preds) / len(preds), 4),
        "unparsed": round(sum(p is None for p in preds) / len(preds), 4),
    }


unpack_outputs()
root = Path(sys.argv[1])
rows = []
for cond_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    for model in MODELS:
        for bench in BENCHES:
            row = summarise(cond_dir, model, bench)
            if row:
                rows.append(row)
write_csv(root / "debiased_results.csv", rows)

index = {(r["model"], r["condition"], r["benchmark"]): r for r in rows}


def cell(model, conds, bench, key):
    vals = [100 * index[(model, c, bench)][key] for c in conds if (model, c, bench) in index]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.1f}"
    return f"{statistics.mean(vals):.1f} ± {statistics.stdev(vals):.1f}"


for bench in BENCHES:
    print(f"\n## {bench} 0-shot: original order vs circular (CIDAR = mean ± std over 3 seeds)\n")
    print("| model | condition | original | circular | all rotations | answers 'A' |\n|---|---|---|---|---|---|")
    for m in MODELS:
        for name, conds in CONDITIONS.items():
            cells = [cell(m, conds, bench, k) for k in ["original", "circular", "all_rot", "pred_A"]]
            print(f"| {m} | {name} | " + " | ".join(cells) + " |")
print(f"\n{len(rows)} rows -> {root / 'debiased_results.csv'}")
