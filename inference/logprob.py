"""Log-probability multiple-choice evaluation on ArabicMMLU and English MMLU.

Same prompts as inference.py, but the answer is the option letter with the highest next-token
probability (as in the ArabicMMLU paper) instead of a parsed greedy generation. So there is no
parsing, no unparsed answers, and no dependence on the format of the reply ('A', 'A.', 'A. صح').

    python logprob.py --model llama8b
    python logprob.py --model qwen7b --benchmarks mmlu --shots 0 --limit 50
    python logprob.py --model llama8b --adapter <adapter dir> --output_dir ../outputs/logprob/lora/<run>
    python logprob.py --model llama8b --circular      # also every rotation of the options (0-shot)

Writes <output_dir>/<benchmark>_<k>shot/<model>.jsonl and <model>_summary.json (circular:
<benchmark>_rot<j>/), in the format of inference.py, so the analysis scripts read them unchanged.
Each record also has `mass` (probability of all option letters, should be near 1) and
`letter_logprobs`. Settings that already have output are skipped.
"""
import argparse
import math
import os
from pathlib import Path

from data import EXTRA_LOADERS, LOADERS, snapshot
from debiased import rotate
from inference import dev_examples
from models import MODELS
from prompts import LETTERS, encode
from results import unpack_outputs, write_results

OUT_DIR = Path(os.environ.get("OUT_DIR", Path(__file__).resolve().parent.parent / "outputs" / "logprob" / "baseline"))


def parse_args():
    p = argparse.ArgumentParser(description="ArabicMMLU / MMLU log-probability evaluation")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--benchmarks", nargs="+", choices=[*LOADERS, *EXTRA_LOADERS], default=list(LOADERS))
    p.add_argument("--shots", nargs="+", type=int, default=[0, 3], help="few-shot examples from the dev split")
    p.add_argument("--circular", action="store_true", help="0-shot, every rotation of the options (instead of --shots)")
    p.add_argument("--output_dir", type=Path, default=OUT_DIR)
    p.add_argument("--adapter", type=Path, default=None, help="LoRA adapter folder from finetune/train_lora.py")
    p.add_argument("--batch_tokens", type=int, default=16384, help="padded tokens per forward pass")
    p.add_argument("--limit", type=int, default=None, help="only the first N questions")
    return p.parse_args()


def letter_ids(tok) -> dict[str, list[int]]:
    """Token ids that can start an answer with each letter: 'A', ' A' and sentencepiece '▁A' variants.

    Tokenizers differ in whether the first reply token carries the space (Mistral), so the probability
    of a letter is summed over all of its variants. Each id is checked to decode back to the letter.
    """
    ids = {}
    for l in LETTERS:
        found = set()
        for piece in (l, "▁" + l, "Ġ" + l):
            i = tok.convert_tokens_to_ids(piece)
            if i is not None and i != tok.unk_token_id and tok.decode([i]).strip() == l:
                found.add(i)
        for text in (l, " " + l):
            enc = tok.encode(text, add_special_tokens=False)
            if len(enc) == 1 and tok.decode(enc).strip() == l:
                found.add(enc[0])
        if not found:
            raise SystemExit(f"no token for option letter {l}")
        ids[l] = sorted(found)
    return ids


def load_model(name: str, adapter: Path | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = str(snapshot(MODELS[name], repo_type="model"))
    tok = AutoTokenizer.from_pretrained(path)
    # gemma-2 needs eager attention for its logit soft-capping (as in training)
    attn = "eager" if name == "gemma2" else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation=attn).cuda()
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    return tok, model.eval()


def score(tok, model, prompts: list[list[int]], ids: dict[str, list[int]], batch_tokens: int) -> list[list[float]]:
    """Per prompt: log-probability of each option letter A-E as the next token (summed over its variants)."""
    import torch

    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)  # longest first
    result = [None] * len(prompts)
    start = 0
    while start < len(order):
        width = len(prompts[order[start]])
        n = max(1, batch_tokens // width)
        batch = order[start: start + n]
        start += n
        input_ids = torch.full((len(batch), width), pad, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, i in enumerate(batch):  # left padding: the last position is the next-token position
            p = prompts[i]
            input_ids[row, width - len(p):] = torch.tensor(p)
            mask[row, width - len(p):] = 1
        position_ids = (mask.cumsum(-1) - 1).clamp(min=0)  # real tokens start at position 0
        with torch.no_grad():
            logits = model(input_ids=input_ids.cuda(), attention_mask=mask.cuda(),
                           position_ids=position_ids.cuda(), logits_to_keep=1).logits[:, -1].float()
        logp = torch.log_softmax(logits, dim=-1)
        per_letter = torch.stack([torch.logsumexp(logp[:, ids[l]], dim=-1) for l in LETTERS], dim=-1).cpu().tolist()
        for row, i in enumerate(batch):
            result[i] = per_letter[row]
    return result


def make_records(rows: list[dict], letter_logp: list[list[float]]) -> list[dict]:
    """One record per question. The prediction is the best letter among the question's own options."""
    records = []
    for r, lp in zip(rows, letter_logp):
        n = len(r["options"])
        pred = LETTERS[max(range(n), key=lambda k: lp[k])]
        records.append({
            "id": r["id"], "task": r["task"], "group": r["group"], "n_options": n,
            "gold": r["gold"], "output": pred, "pred": pred, "correct": pred == r["gold"],
            "mass": round(sum(math.exp(x) for x in lp), 4),
            "letter_logprobs": [round(x, 3) for x in lp[:n]],
        })
    return records


def run(args, tok, model, ids, bench: str, name: str, rows: list[dict], shots: dict | None, k: int, meta: dict) -> None:
    out_path = args.output_dir / f"{bench}_{name}" / f"{args.model}.jsonl"
    if out_path.exists():
        print(f"skip {out_path}", flush=True)
        return
    prompts = [encode(tok, bench, r, shots[r["task"]][:k] if shots else []) for r in rows]
    print(f"{bench} {name}: {len(rows)} questions, longest prompt {max(map(len, prompts))} tokens", flush=True)
    records = make_records(rows, score(tok, model, prompts, ids, args.batch_tokens))
    adapter = str(args.adapter) if args.adapter else None
    write_results(out_path, records, {"model": MODELS[args.model], "adapter": adapter, "benchmark": bench,
                                      "scoring": "letter_logprob", **meta})
    print(f"  mean letter probability mass {sum(r['mass'] for r in records) / len(records):.3f}", flush=True)


def main():
    args = parse_args()
    unpack_outputs()
    tok, model = load_model(args.model, args.adapter)
    ids = letter_ids(tok)
    print("letter token ids:", ids, flush=True)
    for bench in args.benchmarks:
        test = {**LOADERS, **EXTRA_LOADERS}[bench]("test")[: args.limit]
        if args.circular:
            for j in range(len(LETTERS)):
                rows = [rotate(r, j) for r in test if j < len(r["options"])]
                if rows:
                    run(args, tok, model, ids, bench, f"rot{j}", rows, None, 0, {"shots": 0, "rotation": j})
        else:
            for k in args.shots:
                run(args, tok, model, ids, bench, f"{k}shot", test, dev_examples(bench, k) if k else None, k,
                    {"shots": k})


if __name__ == "__main__":
    main()
