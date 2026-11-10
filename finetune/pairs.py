"""Fine-tuning data: CIDAR (Arabic instructions), AlpaGasus (its English source, the control) and
mcq_ar (Arabic multiple-choice questions in the ArabicMMLU prompt format, answer = one letter)."""
import glob
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

DATASETS = {"cidar": "arbml/CIDAR", "alpagasus": "mlabonne/alpagasus", "mcq_ar": "alexandrainst/m_mmlu"}

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))


def load_mcq_ar(folder: str, seed: int) -> list[tuple[str, str]]:
    """(prompt, letter) from the Arabic translation of MMLU (train + val + test, ~14.6k questions, 4 options).

    The prompt is exactly the ArabicMMLU evaluation prompt (inference/prompts.py). Options are shuffled with
    the seed so the answer letter is uniform. Questions that also appear in the ArabicMMLU test set are dropped.
    """
    from data import load_arabicmmlu
    from prompts import LETTERS, user_message

    rng = random.Random(seed)
    norm = lambda s: " ".join(s.split())
    test_questions = {norm(r["question"]) for r in load_arabicmmlu("test")}
    pairs, dropped = {}, 0
    for split in ("train", "val", "test"):
        for line in open(f"{folder}/data/ar/{split}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            options = [r[f"option_{k}"].strip() for k in "abcd"]
            if not r["instruction"].strip() or not all(options) or len(set(options)) < 4:
                continue
            if norm(r["instruction"]) in test_questions:
                dropped += 1
                continue
            gold = options[ord(r["answer"].strip().upper()) - ord("A")]
            rng.shuffle(options)
            row = {"question": r["instruction"].strip(), "context": "", "options": options}
            pairs.setdefault((user_message("arabicmmlu", row), LETTERS[options.index(gold)]), None)
    print(f"mcq_ar: {len(pairs)} questions, {dropped} dropped (in the ArabicMMLU test set)", flush=True)
    return list(pairs)


def load_pairs(name: str, size: int | None, seed: int) -> list[dict]:
    """Unique (prompt, response) pairs in a seeded random order, the first `size` of them, as chat turns."""
    # mcq_ar: only the Arabic files of the multilingual repo are downloaded
    folder = snapshot_download(DATASETS[name], repo_type="dataset", local_files_only=True,
                               allow_patterns=["data/ar/*"] if name == "mcq_ar" else None)
    if name == "mcq_ar":
        pairs = load_mcq_ar(folder, seed)
        random.Random(seed).shuffle(pairs)
        return [{"prompt": [{"role": "user", "content": p}], "completion": [{"role": "assistant", "content": c}]}
                for p, c in pairs[:size]]
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
