"""Build a small reasoning-trace SFT dataset for LoRA verification.

Dataset: ``nvidia/OpenMathReasoning`` (HF). NVIDIA's own curated math
reasoning corpus, used as part of Nemotron 3 post-training. Picking this
dataset matches the model's expected distribution and minimises forgetting
risk during a small LoRA fine-tune.

Why a custom prep step (vs ``datasets`` lib): the vLLM container does not
ship ``datasets``, and the slurm scripts set ``PYTHONNOUSERSITE=1`` so
host-side ``pip install --user datasets`` would not be visible inside the
container. Pure-stdlib over the HF datasets-server REST API works
anywhere with outbound internet (typically the login node), exactly like
``prepare_aime25.py``.

Filter rules (see configs/eval_kaggle.yaml for the constraint):
  - inference_mode == "cot"     keep CoT traces, skip tool-integrated
  - pass_rate_1 >= 1.0          keep traces the teacher solved correctly
  - len(generated_solution) <= ``--max-trace-chars`` (default 14000)
                                ~ 4-4.5K tokens at typical 3 chars/token.
                                Comfortable headroom under the 7680-token
                                eval cap and the 4096-token training
                                ``max_seq_len``.

Output schema per JSONL line:
  - id          (str)            "openmath_<idx>"
  - source_idx  (int)            row offset in the source dataset
  - messages    (list[dict])     OpenAI-style chat list, two turns:
                                     {role: user,      content: <wrapped problem>}
                                     {role: assistant, content: <generated solution>}

Usage (login node):
    python3 scripts/prepare_reasoning_traces.py --num 200
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "nvidia/OpenMathReasoning"
SPLIT = "cot"
PAGE_SIZE = 100  # HF datasets-server max length per call

PROMPT_TEMPLATE = (
    "Solve the following math problem. Make sure to put the answer "
    "(and only answer) inside \\boxed{{}}.\n\n{problem}"
)


def fetch_page(offset: int, length: int) -> list[dict]:
    qs = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": "default",
            "split": SPLIT,
            "offset": offset,
            "length": length,
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{qs}"
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = json.load(resp)
    return [entry["row"] for entry in payload["rows"]]


def keep(row: dict, max_chars: int) -> bool:
    if (row.get("inference_mode") or "").lower() != "cot":
        return False
    pass_rate = row.get("pass_rate_1") or row.get("pass_rate") or 0.0
    try:
        if float(pass_rate) < 1.0:
            return False
    except (TypeError, ValueError):
        return False
    sol = row.get("generated_solution") or ""
    if not sol or len(sol) > max_chars:
        return False
    if "\\boxed{" not in sol:
        return False
    return True


def to_record(row: dict, idx: int) -> dict:
    problem = row["problem"].strip()
    solution = row["generated_solution"].strip()
    return {
        "id": f"openmath_{idx:06d}",
        "source_idx": idx,
        "messages": [
            {"role": "user", "content": PROMPT_TEMPLATE.format(problem=problem)},
            {"role": "assistant", "content": solution},
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=200, help="target sample count after filter")
    ap.add_argument("--max-trace-chars", type=int, default=14000)
    ap.add_argument("--max-pages", type=int, default=200, help="safety cap on pagination")
    ap.add_argument("--output", default="data/lora_traces.jsonl")
    ap.add_argument("--start-offset", type=int, default=0)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept: list[dict] = []
    pages_seen = 0
    offset = args.start_offset

    while len(kept) < args.num and pages_seen < args.max_pages:
        try:
            rows = fetch_page(offset, PAGE_SIZE)
        except Exception as exc:
            print(f"[warn] fetch failed at offset={offset}: {exc}; retrying once")
            rows = fetch_page(offset, PAGE_SIZE)

        if not rows:
            break

        for i, row in enumerate(rows):
            if keep(row, args.max_trace_chars):
                kept.append(to_record(row, offset + i))
                if len(kept) >= args.num:
                    break

        offset += len(rows)
        pages_seen += 1
        print(f"[info] page {pages_seen}: scanned={offset} kept={len(kept)}/{args.num}")

    if not kept:
        raise SystemExit(
            "No samples passed the filter. Inspect the source dataset schema "
            "and adjust keep() — the OpenMathReasoning column names may have "
            "changed since this script was written."
        )

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(kept)} traces -> {out_path}")
    print(f"Scanned {offset - args.start_offset} source rows across {pages_seen} pages.")


if __name__ == "__main__":
    main()
