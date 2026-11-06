"""Instruction datasets for fine-tuning: CIDAR (Arabic) and AlpaGasus (its English source, the control)."""
import glob
import random

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

DATASETS = {"cidar": "arbml/CIDAR", "alpagasus": "mlabonne/alpagasus"}


def load_pairs(name: str, size: int | None, seed: int) -> list[dict]:
    """Unique (prompt, response) pairs in a seeded random order, the first `size` of them, as chat turns."""
    folder = snapshot_download(DATASETS[name], repo_type="dataset", local_files_only=True)
    pairs = {}  # dict keeps file order and drops exact duplicates
    for r in pq.read_table(glob.glob(f"{folder}/data/*.parquet")[0]).to_pylist():
        prompt = r["instruction"].strip()
        if (r.get("input") or "").strip():
            prompt += "\n\n" + r["input"].strip()
        pairs.setdefault((prompt, r["output"].strip()), None)

    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    if size is not None and size > len(pairs):
        raise SystemExit(f"--train_size {size} > {len(pairs)} unique pairs in {name}")
    return [{"prompt": [{"role": "user", "content": p}], "completion": [{"role": "assistant", "content": c}]}
            for p, c in pairs[:size]]
