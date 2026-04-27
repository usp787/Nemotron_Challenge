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
    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
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

    prompts = read_prompts(data_cfg["input_path"])
    print(f"Loaded {len(prompts)} prompts from {data_cfg['input_path']}")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_cfg["name"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        max_model_len=model_cfg.get("max_model_len", 32768),
        gpu_memory_utilization=runtime_cfg.get("gpu_memory_utilization", 0.9),
        tensor_parallel_size=runtime_cfg.get("tensor_parallel_size", 1),
    )

    sampling = SamplingParams(
        temperature=model_cfg.get("temperature", 0.0),
        top_p=model_cfg.get("top_p", 1.0),
        max_tokens=model_cfg.get("max_tokens", 1024),
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    successes = 0
    failures = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for item in prompts:
            record = {
                "id": item.get("id"),
                "prompt": item.get("prompt"),
                "response": None,
                "latency_sec": None,
                "model": model_cfg["name"],
                "backend": runtime_cfg.get("backend", "vllm"),
                "error": None,
            }
            try:
                t0 = time.perf_counter()
                outputs = llm.generate([item["prompt"]], sampling)
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
