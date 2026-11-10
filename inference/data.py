"""Benchmark loaders. Reads the local HF cache only (HF_HOME), never the network."""
import csv
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download, try_to_load_from_cache

from prompts import LETTERS

def snapshot(repo_id: str, repo_type: str = "dataset") -> Path:
    """Local folder of a cached dataset or model (models: folder of config.json, so unused extra files may be missing)."""
    if repo_type == "model":
        config = try_to_load_from_cache(repo_id, "config.json")
        if not isinstance(config, str):
            raise SystemExit(f"{repo_id} is not in the HF cache")
        return Path(config).parent
    return Path(snapshot_download(repo_id, repo_type=repo_type, local_files_only=True))

def load_arabicmmlu(split: str) -> list[dict]:
    """ArabicMMLU rows from the 40 per-task folders (task = folder name, e.g. 'Biology (High School)')."""
    root = snapshot("MBZUAI/ArabicMMLU")
    rows = []
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in ("All", ".git")):
        with open(task_dir / f"{split}.csv", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                options = [r[f"Option {k}"].strip() for k in range(1, 6) if r[f"Option {k}"].strip()]
                gold = r["Answer Key"].strip()
                assert LETTERS.index(gold) < len(options), r["ID"]
                rows.append({
                    "id": f"amm_{r['ID']}", "task": task_dir.name, "group": r["Group"],
                    "question": r["Question"].strip(), "context": r["Context"].strip(),
                    "options": options, "gold": gold,
                })
    return rows

def load_mmlu(split: str) -> list[dict]:
    """English MMLU rows (cais/mmlu 'all' config)."""
    path = next((snapshot("cais/mmlu") / "all").glob(f"{split}-*.parquet"))
    return [
        {"id": f"mmlu_{split}_{i:05d}", "task": r["subject"], "group": None,
         "question": r["question"].strip(), "context": "", "options": list(r["choices"]), "gold": LETTERS[r["answer"]]}
        for i, r in enumerate(pq.read_table(path).to_pylist())
    ]

def load_exams_ar(split: str) -> list[dict]:
    """Arabic EXAMS school-exam questions (OALL/Arabic_EXAMS, 4 options). Used only as a held-out test set."""
    path = next((snapshot("OALL/Arabic_EXAMS") / "data").glob(f"{split}-*.parquet"))
    return [
        {"id": f"exams_{split}_{i:04d}", "task": r["subject"], "group": None, "question": r["question"].strip(),
         "context": "", "options": [r[k].strip() for k in "ABCD"], "gold": r["answer"].strip()}
        for i, r in enumerate(pq.read_table(path).to_pylist())
    ]

LOADERS = {"arabicmmlu": load_arabicmmlu, "mmlu": load_mmlu}
EXTRA_LOADERS = {"exams_ar": load_exams_ar}  # evaluated only when asked for (logprob.py --benchmarks exams_ar)
