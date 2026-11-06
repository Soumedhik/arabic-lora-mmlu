"""The five models and the fine-tuning runs compared in the main tables."""
import os

MODELS = {
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma2": "google/gemma-2-9b-it",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}

# Adapter runs are named <model>_<run>, e.g. llama8b_cidar_r16_a32_s43.
CIDAR_SEEDS = ["cidar_r16_a32", "cidar_r16_a32_s43", "cidar_r16_a32_s44"]
CONDITIONS = {"baseline": ["baseline"], "CIDAR": CIDAR_SEEDS, "AlpaGasus": ["alpagasus_r16_a32"]}


def load_llm(name: str, seed: int, lora: bool = False):
    """Load a cached model into vLLM (bf16, one GPU). lora=True allows adapters up to rank 64."""
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from vllm import LLM

    from data import snapshot

    print(f"Loading {MODELS[name]}", flush=True)
    return LLM(
        model=str(snapshot(MODELS[name], repo_type="model")),
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", 0.80)),
        seed=seed,
        enable_lora=lora,
        max_lora_rank=64,
    )
