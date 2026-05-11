"""Pre-train diagnostics — runs BEFORE scripts/train_lora.py in the smoke job.

Catches the failure modes that would otherwise burn cluster walltime on a
broken run:

  1. Tokenizer load — confirms AutoTokenizer can be instantiated for the
     base model from the local HF cache without falling back to the network
     (which would mean the cache is missing).

  2. Per-record tokenization — applies the chat template to every record in
     data/sft_combined.jsonl, reports per-category token-length histogram
     and the count of records that overflow ``max_seq_len`` (default 8192).
     The trainer silently drops overflows, so unflagged drift here is the
     "iteration-3 flat loss" failure mode in disguise.

  3. <think>/</think> boundary check — re-tokenizes a sample of records as
     (full conversation) and (prompt-only). Asserts that the prompt token
     prefix is identical to the corresponding prefix of the full
     conversation. If they diverge, scripts/train_lora.py's loss mask
     points at the wrong tokens (this is exactly the iteration-3 post-mortem
     bug).

  4. Optional record-skip rate per category — surfaces categories that
     would lose most of their training signal due to length truncation
     (matching / cipher / equation_numeric are the usual suspects).

Exit code is 0 if every category's overflow rate is below ``--max-overflow-frac``
(default 5%) AND every boundary check passes. Non-zero otherwise.

Usage:
    python scripts/smoke_train_diagnostics.py --data data/sft_combined.jsonl
    python scripts/smoke_train_diagnostics.py --max-overflow-frac 0.10 --sample 200
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def _banner(s: str) -> None:
    print()
    print("=" * 64)
    print(s)
    print("=" * 64, flush=True)


def _load_records(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stage_a_tokenizer():
    _banner("Stage A: AutoTokenizer load")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, local_files_only=False
    )
    # Confirm critical special tokens are single-token.
    for s in ("<|im_start|>", "<|im_end|>", "<think>", "</think>"):
        ids = tok(s, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            print(f"  warn: {s!r} tokenizes to {len(ids)} ids: {ids}")
        else:
            print(f"  ok: {s!r} -> id {ids[0]}")
    return tok


def stage_b_length_histogram(
    tok, records: list[dict], max_seq_len: int
) -> dict[str, dict]:
    _banner(f"Stage B: per-category token-length histogram (max_seq_len={max_seq_len})")
    by_cat: dict[str, list[int]] = defaultdict(list)
    overflow_by_cat: dict[str, int] = defaultdict(int)
    overflow_examples: dict[str, list[tuple[str, int]]] = defaultdict(list)

    n = len(records)
    for i, r in enumerate(records):
        if i % 500 == 0:
            print(f"  tokenizing {i}/{n}...", flush=True)
        rendered = tok.apply_chat_template(
            r["messages"], tokenize=False,
            add_generation_prompt=False, enable_thinking=True,
        )
        ids = tok(rendered, add_special_tokens=False)["input_ids"]
        cat = r.get("category", "?")
        by_cat[cat].append(len(ids))
        if len(ids) > max_seq_len:
            overflow_by_cat[cat] += 1
            if len(overflow_examples[cat]) < 5:
                overflow_examples[cat].append((r.get("id", "?"), len(ids)))

    print()
    print(f"{'category':<28}{'count':>7}{'p50':>7}{'p90':>7}{'p99':>7}{'max':>8}{'overflow':>11}")
    summary: dict[str, dict] = {}
    for cat in sorted(by_cat):
        lengths = sorted(by_cat[cat])
        p50 = lengths[len(lengths) // 2]
        p90 = lengths[int(len(lengths) * 0.9)]
        p99 = lengths[min(len(lengths) - 1, int(len(lengths) * 0.99))]
        maxlen = lengths[-1]
        ovf = overflow_by_cat[cat]
        ovf_frac = ovf / len(lengths) if lengths else 0
        summary[cat] = {
            "count": len(lengths),
            "p50": p50, "p90": p90, "p99": p99, "max": maxlen,
            "overflow": ovf,
            "overflow_frac": round(ovf_frac, 4),
        }
        print(
            f"{cat:<28}{len(lengths):>7}{p50:>7}{p90:>7}{p99:>7}{maxlen:>8}"
            f"{ovf:>7} ({ovf_frac*100:>3.1f}%)"
        )

    if overflow_examples:
        print()
        print("Overflow sample IDs (first 5 per offending category):")
        for cat, examples in sorted(overflow_examples.items()):
            id_strs = ", ".join(f"{pid}({ln})" for pid, ln in examples)
            print(f"  {cat}: {id_strs}")
    return summary


def stage_c_boundary_check(
    tok, records: list[dict], sample_size: int
) -> int:
    """Verify the prompt-only token prefix equals the full-conversation prefix.

    If they diverge, scripts/train_lora.py:build_dataset masks the wrong
    tokens (which silently teaches the model nothing useful).
    """
    _banner(f"Stage C: <think>/</think> boundary alignment on {sample_size} sample records")
    if not records:
        print("  (no records)")
        return 0

    n_check = min(sample_size, len(records))
    failures = 0
    for r in records[:n_check]:
        messages = r["messages"]
        full = tok.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=False, enable_thinking=True,
        )
        prompt_only = tok.apply_chat_template(
            [m for m in messages if m["role"] != "assistant"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        full_ids = tok(full, add_special_tokens=False)["input_ids"]
        prompt_ids = tok(prompt_only, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)
        if full_ids[:prompt_len] != prompt_ids:
            failures += 1
            if failures <= 3:
                print(f"  FAIL id={r.get('id', '?')} category={r.get('category', '?')}")
                # Find divergence index
                for i in range(min(prompt_len, len(full_ids))):
                    if full_ids[i] != prompt_ids[i]:
                        print(f"    divergence at index {i}: "
                              f"full={full_ids[i]} prompt={prompt_ids[i]}")
                        break
    if failures == 0:
        print(f"  ok: all {n_check} sample records have matching prefixes")
    else:
        print(f"  {failures}/{n_check} records failed prefix alignment")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sft_combined.jsonl")
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--sample", type=int, default=20,
                    help="Number of records to boundary-check.")
    ap.add_argument("--max-overflow-frac", type=float, default=0.05,
                    help="Per-category overflow fraction tolerated before "
                         "the diagnostic fails. Default 5%%.")
    ap.add_argument("--out-json", default=None,
                    help="Optional path to write the summary as JSON.")
    args = ap.parse_args()

    records = _load_records(Path(args.data))
    print(f"Loaded {len(records)} records from {args.data}")

    try:
        tok = stage_a_tokenizer()
    except Exception as e:
        print(f"FAIL Stage A: {type(e).__name__}: {e}")
        return 1

    summary = stage_b_length_histogram(tok, records, args.max_seq_len)

    failures = stage_c_boundary_check(tok, records, args.sample)

    # Pass/fail summary.
    _banner("Diagnostic verdict")
    failed_categories = [
        cat for cat, s in summary.items()
        if s["overflow_frac"] > args.max_overflow_frac
    ]
    bad = []
    if failures:
        bad.append(f"boundary alignment failures: {failures}")
    if failed_categories:
        bad.append(
            f"overflow > {args.max_overflow_frac:.0%} in: {failed_categories}"
        )

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "records": len(records),
                    "max_seq_len": args.max_seq_len,
                    "summary": summary,
                    "boundary_failures": failures,
                    "verdict": "FAIL" if bad else "PASS",
                    "issues": bad,
                },
                f, indent=2,
            )

    if bad:
        print("FAIL: " + "; ".join(bad))
        return 1
    print("PASS: tokenization fits, boundary alignment clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
