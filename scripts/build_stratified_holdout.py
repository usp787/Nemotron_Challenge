"""Build a stratified multi-category holdout from problems_classified.jsonl.

The previous holdout (data/numeral_holdout.jsonl) was single-category and
gave us no signal on cipher / bit_manipulation / equation_numeric — three
training iterations all scored 100% locally and 0.53-0.57 on Kaggle.

This script samples N rows per category from data/problems_classified.jsonl
(produced by scripts/categorize.py), excluding any prompt already used in
data/sft_traces.jsonl so the holdout is genuinely held out. Output schema
matches what scripts/baseline_generate.py expects: {id, prompt,
expected_answer, category}. The category field is preserved so
scripts/evaluate.py --by-category can report per-category accuracy.

Run on the local box; output is small and gets committed to git so the
cluster receives it via ``git pull``.

Usage:
    python scripts/build_stratified_holdout.py
    python scripts/build_stratified_holdout.py --per-category 100
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classified", default="data/problems_classified.jsonl",
                    help="Source file: output of scripts/categorize.py")
    ap.add_argument("--training", default="data/sft_traces.jsonl",
                    help="Training file used to exclude already-seen prompts")
    ap.add_argument("--output", default="data/stratified_holdout.jsonl")
    ap.add_argument("--per-category", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_prompts: set[str] = set()
    train_path = Path(args.training)
    if train_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                msgs = r.get("messages", [])
                user = next((m["content"] for m in msgs if m.get("role") == "user"), None)
                if user:
                    train_prompts.add(user.strip())
        print(f"[info] training prompts to exclude: {len(train_prompts)}")
    else:
        print(f"[warn] training file not found at {train_path}; not excluding any prompts")

    by_cat: dict[str, list[dict]] = defaultdict(list)
    with open(args.classified, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cat = r.get("category", "unknown")
            if cat == "unknown":
                continue
            if r["prompt"].strip() in train_prompts:
                continue
            by_cat[cat].append(r)

    print(f"[info] available rows after excluding training prompts:")
    for c in sorted(by_cat):
        print(f"  {c:<20} {len(by_cat[c])}")

    rng = random.Random(args.seed)
    out_rows: list[dict] = []
    for cat in sorted(by_cat):
        rows = by_cat[cat]
        if len(rows) < args.per_category:
            print(f"[warn] only {len(rows)} rows for {cat}; using all of them")
            sampled = rows
        else:
            sampled = rng.sample(rows, args.per_category)
        for r in sampled:
            out_rows.append({
                "id": r.get("id"),
                "prompt": r["prompt"],
                "expected_answer": r.get("answer"),
                "category": cat,
            })

    rng.shuffle(out_rows)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[info] wrote {len(out_rows)} rows -> {out_path}")
    counts: dict[str, int] = defaultdict(int)
    for r in out_rows:
        counts[r["category"]] += 1
    print("[info] holdout composition:")
    for c in sorted(counts):
        print(f"  {c:<20} {counts[c]}")


if __name__ == "__main__":
    main()
