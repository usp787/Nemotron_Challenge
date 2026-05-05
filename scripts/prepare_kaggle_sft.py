"""Prepare SFT training and holdout JSONL from the Kaggle train.csv.

The Kaggle competition gives us train.csv with (id, prompt, answer)
triples. We don't have pre-existing reasoning traces, so this script
implements direct-answer SFT (Route A): each training example becomes
prompt -> \\boxed{<answer>}, teaching the model to skip lengthy reasoning
and emit the final answer in the format Kaggle scores on, within the
7680-token greedy decoding budget.

Outputs:
  data/sft_train.jsonl   -- training set: {"messages": [...]}
  data/sft_holdout.jsonl -- local eval set: {"id":..., "prompt":..., "expected_answer":...}

The two slices are disjoint (a single shuffle then take consecutive
blocks under a fixed seed). Pure-stdlib so it runs anywhere with a
modern Python -- laptop, login node, etc -- without extra packages.

Usage:
    python scripts/prepare_kaggle_sft.py --num 3000 --holdout 200
    python scripts/prepare_kaggle_sft.py --train-csv "train&test/train.csv" \\
        --num 5000 --holdout 200 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    """Read train.csv as a list of {id, prompt, answer} dicts.

    csv.DictReader handles multiline quoted prompts (these prompts
    embed newlines inside the quoted field) correctly.
    """
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "prompt", "answer"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"train.csv missing required columns: {sorted(missing)}; "
                f"got: {reader.fieldnames}"
            )
        for row in reader:
            rows.append(row)
    return rows


def write_train_jsonl(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            record = {
                "messages": [
                    {"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": f"\\boxed{{{r['answer']}}}"},
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_holdout_jsonl(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            record = {
                "id": r["id"],
                "prompt": r["prompt"],
                "expected_answer": r["answer"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-csv",
        default="train&test/train.csv",
        help="Path to the Kaggle train.csv (id,prompt,answer).",
    )
    ap.add_argument("--num", type=int, default=3000, help="Training samples")
    ap.add_argument("--holdout", type=int, default=200, help="Local eval samples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-train", default="data/sft_train.jsonl")
    ap.add_argument("--out-holdout", default="data/sft_holdout.jsonl")
    args = ap.parse_args()

    rows = load_rows(Path(args.train_csv))
    print(f"[info] loaded {len(rows)} rows from {args.train_csv}")

    if args.num + args.holdout > len(rows):
        raise SystemExit(
            f"Requested {args.num} train + {args.holdout} holdout = "
            f"{args.num + args.holdout}, but only {len(rows)} rows available."
        )

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    holdout = rows[: args.holdout]
    train = rows[args.holdout : args.holdout + args.num]

    write_holdout_jsonl(holdout, Path(args.out_holdout))
    write_train_jsonl(train, Path(args.out_train))

    print(f"[info] wrote {len(train)} training samples -> {args.out_train}")
    print(f"[info] wrote {len(holdout)} holdout samples -> {args.out_holdout}")


if __name__ == "__main__":
    main()
