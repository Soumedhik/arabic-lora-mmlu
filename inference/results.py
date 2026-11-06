"""Answer parsing and the result files shared by the evaluation and analysis scripts."""
import csv
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from prompts import LETTERS

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"


def unpack_outputs() -> None:
    """The per-question .jsonl files are kept in outputs/raw_outputs.zip. Extract any that are missing."""
    archive = OUTPUTS / "raw_outputs.zip"
    if not archive.exists():
        return
    with zipfile.ZipFile(archive) as z:
        missing = [name for name in z.namelist() if not (OUTPUTS / name).exists()]
        if missing:
            print(f"Unzipping {len(missing)} files from {archive}", flush=True)
            z.extractall(OUTPUTS, members=missing)


def parse_letter(text: str, n_options: int) -> str | None:
    """First standalone option letter in the answer ('B', '**B**', '(B)', 'Answer: B', 'C. <option text>').

    Returns None if there is no letter, or if the answer starts with a list of options
    ('A, B, C', 'A و C', '(A or B)'): a hedge is not a choice, so it counts as wrong.
    """
    letters = LETTERS[:n_options]
    m = re.search(rf"\b([{letters}])\b", text)
    if not m:
        return None
    hedge = re.match(rf"\W*(?:[,،/&]|و|أو|or\b|and\b)\W*[{letters}]\b", text[m.end():])
    return None if hedge else m.group(1)


def make_records(rows: list[dict], outputs) -> list[dict]:
    """One record per question from the vLLM outputs. Unparsed answers count as wrong."""
    records = []
    for r, out in zip(rows, outputs):
        text = out.outputs[0].text
        pred = parse_letter(text, len(r["options"]))
        records.append({
            "id": r["id"], "task": r["task"], "group": r["group"], "n_options": len(r["options"]),
            "gold": r["gold"], "output": text, "pred": pred, "correct": pred == r["gold"],
        })
    return records


def accuracy_by(records: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for r in records:
        groups[r[key]].append(r["correct"])
    return {g: round(sum(v) / len(v), 4) for g, v in sorted(groups.items())}


def write_results(out_path: Path, records: list[dict], meta: dict) -> None:
    """Write <model>.jsonl (one record per question) and <model>_summary.json."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)

    n = len(records)
    summary = {
        **meta,
        "questions": n,
        "accuracy": round(sum(r["correct"] for r in records) / n, 4),
        "unparsed_rate": round(sum(r["pred"] is None for r in records) / n, 4),
        "chance": round(sum(1 / r["n_options"] for r in records) / n, 4),
        "always_A": round(sum(r["gold"] == "A" for r in records) / n, 4),
        "pred_counts": dict(sorted(Counter(str(r["pred"]) for r in records).items())),
        "accuracy_by_group": accuracy_by(records, "group") if meta["benchmark"] == "arabicmmlu" else None,
        "accuracy_by_task": accuracy_by(records, "task"),
    }
    summary_path = out_path.with_name(f"{out_path.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"{meta['benchmark']} {meta['shots']}-shot: accuracy {summary['accuracy']:.4f} "
          f"(unparsed {summary['unparsed_rate']:.4f}) -> {out_path}", flush=True)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
