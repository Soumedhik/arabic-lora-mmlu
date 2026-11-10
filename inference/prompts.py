LETTERS = "ABCDE"

# Instruction in the question's language. Option letters are Latin (A-E) in both, so parsing is the same.
TEMPLATES = {
    "arabicmmlu": "{context}السؤال: {question}\n{options}\n\nأجب بحرف الإجابة الصحيحة فقط ({letters}).",
    "exams_ar": "{context}السؤال: {question}\n{options}\n\nأجب بحرف الإجابة الصحيحة فقط ({letters}).",
    "mmlu": "{context}Question: {question}\n{options}\n\nAnswer with the letter of the correct option only ({letters}).",
}
OR = {"arabicmmlu": " أو ", "exams_ar": " أو ", "mmlu": " or "}


def user_message(bench: str, row: dict) -> str:
    """One question with its lettered options."""
    letters = LETTERS[: len(row["options"])]
    options = "\n".join(f"{l}. {o}" for l, o in zip(letters, row["options"]))
    context = f"{row['context']}\n\n" if row["context"] else ""
    shown = ", ".join(letters[:-1]) + OR[bench] + letters[-1]
    return TEMPLATES[bench].format(context=context, question=row["question"], options=options, letters=shown)


def messages(bench: str, row: dict, shots: list[dict]) -> list[dict]:
    """Chat turns: each few-shot example is a user question + assistant letter, then the test question."""
    msgs = []
    for ex in shots:
        msgs.append({"role": "user", "content": user_message(bench, ex)})
        msgs.append({"role": "assistant", "content": ex["gold"]})
    msgs.append({"role": "user", "content": user_message(bench, row)})
    return msgs


def encode(tok, bench: str, row: dict, shots: list[dict]) -> list[int]:
    """Token ids of the chat prompt. The template already adds BOS, so no extra special tokens."""
    text = tok.apply_chat_template(messages(bench, row, shots), tokenize=False, add_generation_prompt=True)
    return tok.encode(text, add_special_tokens=False)
