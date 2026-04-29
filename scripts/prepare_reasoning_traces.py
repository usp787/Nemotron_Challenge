"""Build a small reasoning-trace SFT dataset for LoRA verification.

Dataset: ``nvidia/OpenMathReasoning`` (HF). NVIDIA's own curated math
reasoning corpus, used as part of Nemotron 3 post-training. Picking this
dataset matches the model's expected distribution and minimises
forgetting risk during a small LoRA fine-tune.

Why a custom prep step (vs ``datasets`` lib): the vLLM container does
not ship ``datasets``, and the slurm scripts set ``PYTHONNOUSERSITE=1``
so host-side ``pip install --user datasets`` would not be visible
inside the container. Pure-stdlib over the HF datasets-server REST API
works anywhere with outbound internet (typically the login node),
exactly like ``prepare_aime25.py``.

Source schema (verified from the HF dataset card):
  problem               str   problem statement (AoPS-derived)
  generated_solution    str   reasoning trace from R1 or QwQ-32B
  expected_answer       str   ground truth (often LaTeX, sometimes int)
  problem_type          str   "has_answer_extracted" |
                              "no_answer_extracted" | "converted_proof"
  inference_mode        str   "cot" | "tir" | "genselect"
  pass_rate_72b_tir     str   pass rate out of 32 by Qwen2.5-Math-72B-TIR,
                              or "n/a" if not evaluated.
  used_in_kaggle        bool  whether AIMO-2 winners used this row.
  generation_model      str   "DeepSeek-R1" | "QwQ-32B"

Filter rules (configurable, see CLI flags):
  - split == "cot"               via the SPLIT constant; we already pull
                                 only the cot split.
  - problem_type == "has_answer_extracted"
                                 skip proof-converted (less reliable)
                                 and no-answer rows (ungradeable).
  - "\\boxed{" in generated_solution
                                 final answer must be boxed for
                                 inference-time format alignment.
  - len(generated_solution) <= ``--max-trace-chars`` (default 14000)
                                 ~ 4-4.5K tokens at typical 3 chars/tok.
                                 Headroom under the 7680-token eval cap
                                 and the 4096-token training max_seq_len.
  - --min-pass-rate (default 0.0, optional)
                                 if set > 0, requires pass_rate_72b_tir
                                 numeric value >= threshold; rows with
                                 "n/a" are kept by default since most R1
                                 traces aren't 72B-TIR-evaluated and are
                                 still high quality.

Output schema per JSONL line:
  - id          (str)            "openmath_<idx>"
  - source_idx  (int)            row offset in the source dataset
  - messages    (list[dict])     OpenAI-style chat list, two turns:
                                     {role: user, content: <wrapped problem>}
                                     {role: assistant, content: <trace>}

Usage (login node):
    python3 scripts/prepare_reasoning_traces.py --num 200
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "nvidia/OpenMathReasoning"
SPLIT = "cot"
PAGE_SIZE = 100  # HF datasets-server max length per call
USER_AGENT = "Nemotron_Challenge/lora-verification (educational; Kaggle)"

PROMPT_TEMPLATE = (
    "Solve the following math problem. Make sure to put the answer "
    "(and only answer) inside \\boxed{{}}.\n\n{problem}"
)


def fetch_page(offset: int, length: int, max_retries: int = 5) -> list[dict]:
    """Fetch a page of rows. Backs off on 429 by Retry-After or exponential."""
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

    delay = 5.0
    for attempt in range(max_retries):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.load(resp)
            return [entry["row"] for entry in payload["rows"]]
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                ra = exc.headers.get("Retry-After")
                wait = float(ra) if ra and ra.isdigit() else delay
                print(f"[warn] 429 at offset={offset}; sleeping {wait:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                delay = min(delay * 2, 120.0)
                continue
            raise
        except Exception as exc:
            if attempt + 1 == max_retries:
                raise
            print(f"[warn] fetch error at offset={offset}: {exc}; retry in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 120.0)
    raise RuntimeError(f"fetch_page exhausted retries at offset={offset}")


def parse_pass_rate(value) -> float | None:
    """Return float pass-rate or None for missing/'n/a'."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s or s == "n/a":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def keep(row: dict, max_chars: int, min_pass_rate: float) -> bool:
    if (row.get("inference_mode") or "").lower() != "cot":
        return False
    if (row.get("problem_type") or "") != "has_answer_extracted":
        return False
    sol = row.get("generated_solution") or ""
    if not sol or len(sol) > max_chars:
        return False
    if "\\boxed{" not in sol:
        return False
    if not (row.get("problem") or "").strip():
        return False
    if min_pass_rate > 0.0:
        pr = parse_pass_rate(row.get("pass_rate_72b_tir"))
        if pr is None or pr < min_pass_rate:
            return False
    return True


def reject_reason(row: dict, max_chars: int, min_pass_rate: float) -> str:
    """Return a short tag explaining why a row was rejected (for stats)."""
    if (row.get("inference_mode") or "").lower() != "cot":
        return "not_cot"
    if (row.get("problem_type") or "") != "has_answer_extracted":
        return f"problem_type:{row.get('problem_type', 'missing')}"
    sol = row.get("generated_solution") or ""
    if not sol:
        return "empty_solution"
    if len(sol) > max_chars:
        return "trace_too_long"
    if "\\boxed{" not in sol:
        return "no_boxed"
    if not (row.get("problem") or "").strip():
        return "empty_problem"
    if min_pass_rate > 0.0:
        pr = parse_pass_rate(row.get("pass_rate_72b_tir"))
        if pr is None:
            return "no_pass_rate"
        if pr < min_pass_rate:
            return "low_pass_rate"
    return "kept"


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
    ap.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.0,
        help="if > 0, drop rows whose pass_rate_72b_tir is missing or below this. "
        "Default 0.0 keeps n/a rows (most R1 traces are not 72B-TIR-evaluated).",
    )
    ap.add_argument(
        "--page-sleep",
        type=float,
        default=0.5,
        help="seconds to sleep between page requests (politeness, avoids 429).",
    )
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept: list[dict] = []
    rejects: dict[str, int] = {}
    pages_seen = 0
    offset = args.start_offset

    while len(kept) < args.num and pages_seen < args.max_pages:
        rows = fetch_page(offset, PAGE_SIZE)

        if pages_seen == 0 and rows:
            # Make schema drift obvious on the first page rather than after
            # 4000+ silent rejections.
            sample = rows[0]
            sample_sol = sample.get("generated_solution") or ""
            boxed_marker = "\\boxed{"
            has_boxed = boxed_marker in sample_sol
            print(f"[diag] first row keys: {sorted(sample.keys())}")
            print(
                f"[diag] first row sample: "
                f"problem_type={sample.get('problem_type')!r} "
                f"inference_mode={sample.get('inference_mode')!r} "
                f"pass_rate_72b_tir={sample.get('pass_rate_72b_tir')!r} "
                f"len(generated_solution)={len(sample_sol)} "
                f"has_boxed={has_boxed}"
            )

        if not rows:
            break

        for i, row in enumerate(rows):
            if keep(row, args.max_trace_chars, args.min_pass_rate):
                kept.append(to_record(row, offset + i))
                if len(kept) >= args.num:
                    break
            else:
                tag = reject_reason(row, args.max_trace_chars, args.min_pass_rate)
                rejects[tag] = rejects.get(tag, 0) + 1

        offset += len(rows)
        pages_seen += 1
        print(f"[info] page {pages_seen}: scanned={offset} kept={len(kept)}/{args.num}")

        if len(kept) < args.num and args.page_sleep > 0:
            time.sleep(args.page_sleep)

    if not kept:
        print("[diag] reject tally:")
        for tag, n in sorted(rejects.items(), key=lambda kv: -kv[1]):
            print(f"  {tag}: {n}")
        raise SystemExit(
            "No samples passed the filter. Inspect the [diag] lines above — "
            "the most common reject tag points at the failing condition. "
            "If schema drifted, update keep() to match."
        )

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(kept)} traces -> {out_path}")
    print(f"Scanned {offset - args.start_offset} source rows across {pages_seen} pages.")
    if rejects:
        print("Reject tally:")
        for tag, n in sorted(rejects.items(), key=lambda kv: -kv[1]):
            print(f"  {tag}: {n}")


if __name__ == "__main__":
    main()
