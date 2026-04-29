"""Baseline generation driver.

Reads a YAML config and a JSONL prompt file, runs the configured
inference backend (currently vLLM), and writes one JSON record per
prompt with response text, latency, and error fields.

A failure on a single prompt is recorded in that prompt's record and
does not stop the run.
"""
import argparse
import json
import time
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_prompts(path: str) -> list[dict]:
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Input JSONL not found: {prompt_path}. "
            "Smoke runs use data/sample_prompts_5.jsonl (tracked in git). "
            "AIME25 baseline runs require data/aime25.jsonl - generate it "
            "with `python3 scripts/prepare_aime25.py` from a node with "
            "outbound internet (typically the login node)."
        )

    items: list[dict] = []
    with open(prompt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_record(fp, record: dict) -> None:
    fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    fp.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output", required=True, help="Path to output JSONL")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    runtime_cfg = cfg["runtime"]
    data_cfg = cfg["data"]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = read_prompts(data_cfg["input_path"])
    print(f"Loaded {len(prompts)} prompts from {data_cfg['input_path']}")

    from vllm import LLM, SamplingParams

    # Optional LoRA adapter. When runtime.lora.enabled is true and a path is
    # given, vLLM is initialised with enable_lora=True and each generate()
    # call passes a LoRARequest. This mirrors the Kaggle eval harness, which
    # loads the user's adapter onto the same base model via vLLM.
    lora_runtime = (runtime_cfg.get("lora") or {})
    lora_enabled = bool(lora_runtime.get("enabled")) and bool(lora_runtime.get("path"))
    lora_request = None

    llm_kwargs = dict(
        model=model_cfg["name"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        max_model_len=model_cfg.get("max_model_len", 32768),
        gpu_memory_utilization=runtime_cfg.get("gpu_memory_utilization", 0.9),
        tensor_parallel_size=runtime_cfg.get("tensor_parallel_size", 1),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    if "max_num_seqs" in runtime_cfg:
        llm_kwargs["max_num_seqs"] = runtime_cfg["max_num_seqs"]
    if lora_enabled:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = int(lora_runtime.get("max_lora_rank", 32))
        llm_kwargs["max_loras"] = int(lora_runtime.get("max_loras", 1))

    llm = LLM(**llm_kwargs)

    if lora_enabled:
        from vllm.lora.request import LoRARequest

        adapter_path = str(Path(lora_runtime["path"]).resolve())
        lora_request = LoRARequest("submission_adapter", 1, adapter_path)
        print(f"LoRA adapter loaded: {adapter_path}")

    sampling = SamplingParams(
        temperature=model_cfg.get("temperature", 0.0),
        top_p=model_cfg.get("top_p", 1.0),
        max_tokens=model_cfg.get("max_tokens", 1024),
    )

    # Nemotron 3 Nano relies on its chat template to inject the <think>
    # reasoning prefix. Sending the raw user prompt to llm.generate()
    # bypasses that and silently disables thinking mode, so we apply the
    # template explicitly with enable_thinking=True (the model card's
    # default and what the published AIME25 numbers were produced with).
    tokenizer = llm.get_tokenizer()

    successes = 0
    failures = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for item in prompts:
            formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": item["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            record = {
                "id": item.get("id"),
                "prompt": item.get("prompt"),
                "formatted_prompt": formatted_prompt,
                "expected_answer": item.get("expected_answer"),
                "response": None,
                "latency_sec": None,
                "model": model_cfg["name"],
                "backend": runtime_cfg.get("backend", "vllm"),
                "error": None,
            }
            try:
                t0 = time.perf_counter()
                gen_kwargs = {}
                if lora_request is not None:
                    gen_kwargs["lora_request"] = lora_request
                outputs = llm.generate([formatted_prompt], sampling, **gen_kwargs)
                record["latency_sec"] = round(time.perf_counter() - t0, 3)
                record["response"] = outputs[0].outputs[0].text
                successes += 1
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                failures += 1
            write_record(out, record)

    print(f"Done. successes={successes} failures={failures} -> {out_path}")


if __name__ == "__main__":
    main()
