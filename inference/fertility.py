"""Tokenizer fertility (tokens per word) of each model on ArabicMMLU vs English MMLU test questions.

    python fertility.py         # writes ../outputs/fertility.csv and prints a table

Text per question = context + question + options; words are whitespace-separated; no special tokens.
The two benchmarks are not translations of each other, so the Arabic/English ratio is indicative only.
"""
import os
from pathlib import Path

from transformers import AutoTokenizer

from data import LOADERS, snapshot
from models import MODELS
from results import write_csv

OUT_DIR = Path(os.environ.get("OUT_DIR", Path(__file__).resolve().parent.parent / "outputs"))


def texts(bench: str) -> list[str]:
    return [" ".join([r["context"], r["question"], *r["options"]]).strip() for r in LOADERS[bench]("test")]


def main():
    data = {bench: texts(bench) for bench in LOADERS}
    words = {bench: sum(len(t.split()) for t in ts) for bench, ts in data.items()}
    rows = []
    for name, repo in MODELS.items():
        tok = AutoTokenizer.from_pretrained(snapshot(repo, repo_type="model"))
        row = {"model": name, "vocab_size": len(tok)}
        for bench, ts in data.items():
            tokens = sum(len(ids) for ids in tok(ts, add_special_tokens=False)["input_ids"])
            row[f"{bench}_tokens_per_word"] = round(tokens / words[bench], 3)
        row["ar_en_ratio"] = round(row["arabicmmlu_tokens_per_word"] / row["mmlu_tokens_per_word"], 3)
        rows.append(row)
        print(row, flush=True)

    out = OUT_DIR / "fertility.csv"
    write_csv(out, rows)
    print(f"\nwords: {words}\n")
    print("| model | vocab | Arabic tokens/word | English tokens/word | Ar/En |\n|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['vocab_size']} | {r['arabicmmlu_tokens_per_word']} | "
              f"{r['mmlu_tokens_per_word']} | {r['ar_en_ratio']} |")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
