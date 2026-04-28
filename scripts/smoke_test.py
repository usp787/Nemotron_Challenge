"""Staged smoke test for the Nemotron baseline pipeline.

Implements stages 2 through 5 of the staged smoke test described in
README section 8.3. Stages 0 and 1 (Slurm allocation, Apptainer --nv
GPU passthrough) are handled in slurm/smoke_test.slurm before this
script runs. Stage 6 (5-prompt mini baseline) is handled by
scripts/baseline_generate.py with configs/smoke_h200.yaml.

  Stage 2: Python package check (torch, transformers, vllm)
  Stage 3: Hugging Face metadata access (no weight download)
  Stage 4: Tokenizer + chat template load
  Stage 5: Single-prompt model load and generation

Each stage is run independently. A stage failure is recorded but does
not stop later stages, so a single run produces a complete diagnosis
of which layer is broken.

Exit code is 0 if all stages pass, 1 otherwise.
"""
import sys
import time
import traceback

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

# Stage 5 settings — kept aligned with configs/smoke_h200.yaml.
SMOKE_MAX_MODEL_LEN = 8192
SMOKE_MAX_TOKENS = 64
SMOKE_GPU_MEMORY_UTIL = 0.80


def _banner(title: str) -> None:
    print()
    print("=" * 64)
    print(title)
    print("=" * 64, flush=True)


def stage_2_packages() -> None:
    _banner("Stage 2: Python package check")
    import torch

    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("gpu count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("gpu name:", torch.cuda.get_device_name(0))
        print("bf16 supported:", torch.cuda.is_bf16_supported())
    else:
        raise RuntimeError("CUDA not available inside container")

    import transformers

    print("transformers:", transformers.__version__)

    import vllm

    print("vllm:", getattr(vllm, "__version__", "unknown"))


def stage_3_hf_metadata() -> None:
    _banner("Stage 3: Hugging Face metadata access")
    from huggingface_hub import model_info

    info = model_info(MODEL_ID)
    print("model id:", info.modelId)
    print("sha:", info.sha)


def stage_4_tokenizer() -> None:
    _banner("Stage 4: Tokenizer + chat template")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    messages = [
        {"role": "user", "content": "Answer in one sentence: what is 2 + 2?"}
    ]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(text[:500])


def stage_5_generation() -> None:
    _banner("Stage 5: Single-prompt model load and generation")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        max_model_len=SMOKE_MAX_MODEL_LEN,
        gpu_memory_utilization=SMOKE_GPU_MEMORY_UTIL,
        tensor_parallel_size=1,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=SMOKE_MAX_TOKENS
    )

    t0 = time.perf_counter()
    out = llm.generate(["Answer in one sentence: what is 2 + 2?"], sampling)
    latency = time.perf_counter() - t0
    text = out[0].outputs[0].text
    print(f"latency_sec: {latency:.3f}")
    print(f"response: {text!r}")


def main() -> int:
    stages = [
        ("stage_2_packages", stage_2_packages),
        ("stage_3_hf_metadata", stage_3_hf_metadata),
        ("stage_4_tokenizer", stage_4_tokenizer),
        ("stage_5_generation", stage_5_generation),
    ]

    failed: list[str] = []
    for name, fn in stages:
        try:
            fn()
        except Exception:
            failed.append(name)
            print(f"FAILED: {name}", file=sys.stderr)
            traceback.print_exc()

    _banner("Summary")
    if failed:
        print("FAILED stages:", ", ".join(failed))
        return 1
    print("All stages PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
