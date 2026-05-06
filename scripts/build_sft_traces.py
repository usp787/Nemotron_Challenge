"""Generate SFT training traces and per-category holdout JSONLs.

Reads ``data/problems_classified.jsonl``, dispatches each problem to
the matching reasoner, verifies the trace's ``\\boxed{}`` answer
against ground truth, and writes:

  data/sft_traces.jsonl              (training, ``messages`` format)
  data/<category>_holdout.jsonl     (per-category eval set, one file each)

Wrong-answer traces are dropped; categories without a registered
generator skip trace generation but still get a holdout JSONL written
(so paired base-vs-LoRA eval can run on them once we add the solver).

Mirrors the official Kaggle ``compare_answer`` semantics:
  - pure binary string -> exact lowercase compare
  - else if both float-parseable -> math.isclose(rel_tol=1e-2, abs_tol=1e-5)
  - else -> case-insensitive string compare

Usage:
    python scripts/build_sft_traces.py
    python scripts/build_sft_traces.py --only-category numeral --holdout 200
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

# Make `from reasoners import ...` work when run as
# `python scripts/build_sft_traces.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reasoners import GENERATORS  # noqa: E402
from reasoners.store_types import build_problem  # noqa: E402


_BOXED_RE = re.compile(r"\\boxed\{([^}]*)(?:\}|$)")


def extract_boxed(text: str) -> str:
    matches = _BOXED_RE.findall(text)
    if not matches:
        return ""
    non_empty = [m.strip() for m in matches if m.strip()]
    if non_empty:
        return non_empty[-1]
    return matches[-1].strip()


def restructure_for_thinking(trace: str, answer: str) -> str:
    """Reshape a trace as a complete `<think>...</think>\\boxed{}` assistant turn.

    Nemotron 3 Nano's chat template (audited against the live tokenizer
    config) does *not* auto-prepend ``<think>`` when an assistant message
    already contains ``</think>`` -- the content emits verbatim. So the
    SFT data itself has to include the opening tag, otherwise the
    full-text render and the inference prompt prefix don't share a
    common prefix and ``train_lora.py``'s prompt-length-boundary mask
    points at the wrong token index.

    Resulting assistant content:

        <think>
        <reasoning body>
        </think>

        \\boxed{<answer>}

    With this shape, the inference prompt (which ends in
    ``<|im_start|>assistant\\n<think>\\n``) is a true prefix of the
    training render, so masking and supervision align cleanly.
    """
    lines = trace.splitlines()
    while lines and ("\\boxed{" in lines[-1] or not lines[-1].strip()):
        lines.pop()
    body = "\n".join(lines).rstrip()
    return f"<think>\n{body}\n</think>\n\n\\boxed{{{answer}}}"


def compare_answer(stored: str, predicted: str) -> bool:
    stored = stored.strip()
    predicted = predicted.strip()
    if re.fullmatch(r"[01]+", stored):
        return predicted.lower() == stored.lower()
    try:
        return math.isclose(
            float(stored), float(predicted), rel_tol=1e-2, abs_tol=1e-5
        )
    except Exception:
        return predicted.lower() == stored.lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classified", default="data/problems_classified.jsonl")
    ap.add_argument("--out-train", default="data/sft_traces.jsonl")
    ap.add_argument("--out-holdout-dir", default="data")
    ap.add_argument("--holdout", type=int, default=200,
                    help="Per-category holdout size")
    ap.add_argument("--max-train-per-category", type=int, default=None,
                    help="Cap training samples per category (after holdout)")
    ap.add_argument("--only-category", default=None,
                    help="If set, process only this category")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no-thinking-wrap",
        action="store_true",
        help="Emit the raw trace verbatim (legacy shape). Default wraps "
             "the trace as the post-`<think>` portion of the assistant turn, "
             "matching Nemotron's enable_thinking=True chat template.",
    )
    args = ap.parse_args()

    rows: list[dict] = []
    with open(args.classified, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    by_cat: dict[str, list[dict]] = {}
    for row in rows:
        by_cat.setdefault(row["category"], []).append(row)

    print(f"Loaded {len(rows)} classified problems across {len(by_cat)} categories:")
    for cat, items in sorted(by_cat.items()):
        gen_marker = "(generator)" if cat in GENERATORS else "(no generator yet)"
        print(f"  {cat:30} {len(items):>6}  {gen_marker}")
    print()

    cats_to_process = (
        [args.only_category] if args.only_category else sorted(by_cat.keys())
    )

    rng = random.Random(args.seed)
    train_records: list[dict] = []
    rule_found_total = 0
    rule_unknown_total = 0
    wrong_answer_total = 0
    no_generator_total = 0

    for cat in cats_to_process:
        items = by_cat.get(cat, [])
        if not items:
            print(f"[{cat}] no rows in classified file, skipping")
            continue

        # Deterministic per-category shuffle so train/holdout split is stable
        # across runs and disjoint across categories.
        local_rng = random.Random(args.seed + (hash(cat) & 0xFFFFFFFF))
        items_shuffled = list(items)
        local_rng.shuffle(items_shuffled)

        holdout_n = min(args.holdout, len(items_shuffled))
        holdout_items = items_shuffled[:holdout_n]
        train_items = items_shuffled[holdout_n:]
        if args.max_train_per_category is not None:
            train_items = train_items[: args.max_train_per_category]

        # Always write the holdout, even if no generator yet.
        holdout_path = Path(args.out_holdout_dir) / f"{cat}_holdout.jsonl"
        holdout_path.parent.mkdir(parents=True, exist_ok=True)
        with open(holdout_path, "w", encoding="utf-8") as f:
            for r in holdout_items:
                f.write(json.dumps({
                    "id": r["id"],
                    "prompt": r["prompt"],
                    "expected_answer": r["answer"],
                }, ensure_ascii=False) + "\n")
        print(f"[{cat}] holdout {len(holdout_items):>5} -> {holdout_path}")

        gen = GENERATORS.get(cat)
        if gen is None:
            no_generator_total += len(train_items)
            print(f"[{cat}] no generator -- skipping trace generation "
                  f"({len(train_items)} would-be train rows)")
            continue

        cat_found = 0
        cat_unknown = 0
        cat_wrong = 0
        for r in train_items:
            problem = build_problem(r)
            try:
                trace = gen(problem)
            except Exception as e:
                print(f"[{cat}] {r['id']}: generator error: {type(e).__name__}: {e}")
                trace = None
            if trace is None:
                cat_unknown += 1
                continue
            answer = extract_boxed(trace)
            if not compare_answer(problem.answer, answer):
                cat_wrong += 1
                continue
            content = (
                trace if args.no_thinking_wrap
                else restructure_for_thinking(trace, answer)
            )
            train_records.append({
                "messages": [
                    {"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": content},
                ]
            })
            cat_found += 1

        rule_found_total += cat_found
        rule_unknown_total += cat_unknown
        wrong_answer_total += cat_wrong
        denom = max(1, len(train_items))
        print(
            f"[{cat}] rule_found {cat_found}/{len(train_items)} "
            f"({100.0 * cat_found / denom:.1f}%); "
            f"unknown {cat_unknown}, wrong {cat_wrong}"
        )

    # Interleave categories so SFT batches see a mix.
    rng.shuffle(train_records)

    out_train = Path(args.out_train)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    with open(out_train, "w", encoding="utf-8") as f:
        for rec in train_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print()
    print(f"Total training traces: {rule_found_total} -> {out_train}")
    print(f"Skipped (no generator): {no_generator_total}")
    print(f"Skipped (rule_unknown): {rule_unknown_total}")
    print(f"Skipped (wrong answer): {wrong_answer_total}")


if __name__ == "__main__":
    main()
