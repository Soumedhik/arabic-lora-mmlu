"""Subject-level error analysis of the log-probability baselines on ArabicMMLU (no GPU needed).

    python subject_analysis.py ../outputs --shots 0     # writes <outputs>/subject_analysis_0shot.csv

Reads <outputs>/logprob/baseline/arabicmmlu_<k>shot/<model>.jsonl (and, if present, the circular
arabicmmlu_rot<j> folders). Prints, over the 5 models:
  1. accuracy per task next to chance, hardest to easiest
  2. accuracy by school level and by subject area
  3. consensus: questions every model gets right / wrong, and tasks where the models disagree
  4. position bias per task: how much more often a model answers 'A' than the gold answer is 'A'
  5. circular accuracy per task (only for models whose rotations are all there)
"""
import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path

from models import MODELS
from results import read_jsonl, write_csv

p = argparse.ArgumentParser()
p.add_argument("outputs", type=Path)
p.add_argument("--shots", type=int, default=0)
p.add_argument("--top", type=int, default=8, help="rows in the hardest/easiest/bias lists")
args = p.parse_args()
BASE = args.outputs / "logprob" / "baseline"


def level_of(task: str) -> str:
    m = re.match(r"^.*\((.*)\)$", task)
    return m.group(1) if m else "(none)"


def subject_of(task: str) -> str:
    return re.sub(r"\s*\(.*\)$", "", task)


recs = {m: {r["id"]: r for r in read_jsonl(BASE / f"arabicmmlu_{args.shots}shot" / f"{m}.jsonl")} for m in MODELS}
ids = list(recs["llama8b"])
assert all(list(recs[m]) == ids for m in MODELS), "models answered different questions"
meta = {i: recs["llama8b"][i] for i in ids}  # task, group, n_options, gold are the same for every model


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def acc(m, qs):
    return 100 * mean(recs[m][i]["correct"] for i in qs)


tasks = defaultdict(list)
for i in ids:
    tasks[meta[i]["task"]].append(i)

# ---- 1. per task
rows = []
for t, qs in tasks.items():
    row = {"task": t, "group": meta[qs[0]]["group"], "level": level_of(t), "n": len(qs),
           "chance": round(100 * mean(1 / meta[i]["n_options"] for i in qs), 1)}
    for m in MODELS:
        row[m] = round(acc(m, qs), 1)
    row["mean"] = round(mean(row[m] for m in MODELS), 1)
    row["mean_minus_chance"] = round(row["mean"] - row["chance"], 1)
    row["model_std"] = round(statistics.pstdev(row[m] for m in MODELS), 1)
    row["all_right"] = round(100 * mean(all(recs[m][i]["correct"] for m in MODELS) for i in qs), 1)
    row["all_wrong"] = round(100 * mean(not any(recs[m][i]["correct"] for m in MODELS) for i in qs), 1)
    for m in MODELS:  # position bias: share of answers that are A minus share of gold answers that are A
        row[f"biasA_{m}"] = round(100 * (mean(recs[m][i]["pred"] == "A" for i in qs)
                                         - mean(meta[i]["gold"] == "A" for i in qs)), 1)
    rows.append(row)
rows.sort(key=lambda r: r["mean_minus_chance"])


def table(title, header, body):
    print(f"\n## {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for b in body:
        print("| " + " | ".join(str(x) for x in b) + " |")


cols = ["task", "n", "chance", *MODELS, "mean", "mean−chance"]
as_row = lambda r: [r["task"], r["n"], r["chance"], *[r[m] for m in MODELS], r["mean"], r["mean_minus_chance"]]
table(f"ArabicMMLU {args.shots}-shot: hardest tasks (mean over models minus chance)", cols, map(as_row, rows[: args.top]))
table("Easiest tasks", cols, map(as_row, rows[::-1][: args.top]))
weak = [r for r in rows if r["mean_minus_chance"] < 10]
print(f"\nTasks where the average model is under 10 points above chance: {len(weak)} of {len(rows)} "
      f"({sum(r['n'] for r in weak)} questions): " + ", ".join(r["task"] for r in weak))

# ---- 2. level and subject area
for title, key in (("school level", lambda t: level_of(t)), ("subject area (task name)", lambda t: subject_of(t)),
                   ("group", lambda t: meta[tasks[t][0]]["group"])):
    pooled = defaultdict(list)
    for t, qs in tasks.items():
        pooled[key(t)] += qs
    body = sorted(([k, len(qs), *[f"{acc(m, qs):.1f}" for m in MODELS],
                    f"{mean(acc(m, qs) for m in MODELS):.1f}"] for k, qs in pooled.items()), key=lambda r: float(r[-1]))
    table(f"Accuracy by {title}", [title, "n", *MODELS, "mean"], body)

# ---- 3. consensus
n = len(ids)
right = sum(all(recs[m][i]["correct"] for m in MODELS) for i in ids)
wrong = sum(not any(recs[m][i]["correct"] for m in MODELS) for i in ids)
best = sum(any(recs[m][i]["correct"] for m in MODELS) for i in ids)
print(f"\n## Consensus over {len(MODELS)} models\n\nall right {100 * right / n:.1f}%, all wrong {100 * wrong / n:.1f}%, "
      f"at least one right {100 * best / n:.1f}% (oracle ceiling)")
by_std = sorted(rows, key=lambda r: -r["model_std"])[: args.top]
table("Tasks where the models disagree most (std of accuracy across models)", ["task", *MODELS, "std"],
      [[r["task"], *[r[m] for m in MODELS], r["model_std"]] for r in by_std])
by_wrong = sorted(rows, key=lambda r: -r["all_wrong"])[: args.top]
table("Tasks with most questions no model gets right", ["task", "n", "all wrong %", "all right %"],
      [[r["task"], r["n"], r["all_wrong"], r["all_right"]] for r in by_wrong])

# ---- 4. position bias
table("Position bias: pred-A rate minus gold-A rate, pooled over tasks (points)", ["model", "bias toward A"],
      [[m, f"{100 * (mean(recs[m][i]['pred'] == 'A' for i in ids) - mean(meta[i]['gold'] == 'A' for i in ids)):+.1f}"]
       for m in MODELS])
for m in ("llama8b", "mistral"):
    worst = sorted(rows, key=lambda r: -r[f"biasA_{m}"])[: args.top // 2]
    table(f"{m}: tasks with the strongest pull toward A", ["task", "bias toward A", "accuracy"],
          [[r["task"], f"{r[f'biasA_{m}']:+.1f}", r[m]] for r in worst])

# ---- 5. circular
circ = {}
for m in MODELS:
    files = sorted(BASE.glob(f"arabicmmlu_rot*/{m}.jsonl"))
    if not files:
        continue
    per_q = defaultdict(list)
    for f in files:
        for r in read_jsonl(f):
            per_q[r["id"]].append(r["correct"])
    if sum(len(v) == meta[i]["n_options"] for i, v in per_q.items() if i in meta) == n:
        circ[m] = {i: mean(v) for i, v in per_q.items()}
if circ:
    body = []
    for r in rows:
        qs = tasks[r["task"]]
        cells = []
        for m in circ:
            c = 100 * mean(circ[m][i] for i in qs)
            r[f"circular_{m}"] = round(c, 1)
            cells.append(f"{r[m]:.1f} → {c:.1f}")
        body.append([r["task"], *cells])
    table("Original → circular accuracy per task (hardest first)", ["task", *circ], body)
    print("\nOverall original → circular: " + ", ".join(
        f"{m} {acc(m, ids):.1f} → {100 * mean(circ[m][i] for i in ids):.1f}" for m in circ))
else:
    print("\n(no complete circular runs yet: rerun this script when they finish)")

out = args.outputs / f"subject_analysis_{args.shots}shot.csv"
write_csv(out, rows)
print(f"\n-> {out}")
