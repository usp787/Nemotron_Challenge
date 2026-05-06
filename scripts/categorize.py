"""Classify each train.csv row by Kaggle category.

The Kaggle prompts use a deterministic templated header per category.
A small regex table is sufficient. Output is one JSONL record per row
with {id, prompt, answer, category}; ``unknown`` is emitted only if
no rule matched (audit by tail-checking the count for that bucket).

Run on the local box; output is small and gets committed to git so
the cluster receives it via ``git pull``.

Usage:
    python scripts/categorize.py
    python scripts/categorize.py --train-csv "train&test/train.csv"
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path


# Header-line regex per category. Rule order doesn't matter (each is
# uniquely identifying), but the list documents all categories we see
# in train.csv. Phrases verified by audit (1576-1602 hits per pattern,
# 9500 total).
CATEGORY_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("bit_manipulation", re.compile(r"bit manipulation rule transforms 8-bit binary numbers")),
    ("cipher",            re.compile(r"secret encryption rules are used on text")),
    ("numeral",           re.compile(r"numbers are secretly converted into a different numeral system")),
    ("unit_conversion",   re.compile(r"a secret unit conversion is applied to measurements")),
    ("gravity",           re.compile(r"the gravitational constant has been secretly changed")),
    ("equation_numeric",  re.compile(r"a secret set of transformation rules is applied to equations")),
]


def classify(prompt: str) -> str:
    for name, pat in CATEGORY_RULES:
        if pat.search(prompt):
            return name
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="train&test/train.csv")
    ap.add_argument("--out", default="data/problems_classified.jsonl")
    args = ap.parse_args()

    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    counts: Counter[str] = Counter()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.train_csv, "r", encoding="utf-8", newline="") as f, \
         open(out_path, "w", encoding="utf-8") as out:
        reader = csv.DictReader(f)
        for row in reader:
            cat = classify(row["prompt"])
            counts[cat] += 1
            out.write(json.dumps({
                "id": row["id"],
                "prompt": row["prompt"],
                "answer": row["answer"],
                "category": cat,
            }, ensure_ascii=False) + "\n")

    total = sum(counts.values())
    print(f"Classified {total} rows -> {out_path}")
    for cat, n in counts.most_common():
        pct = 100.0 * n / total if total else 0.0
        print(f"  {cat:30} {n:>6} ({pct:.1f}%)")
    if counts.get("unknown", 0):
        print()
        print(f"WARNING: {counts['unknown']} rows did not match any category rule.")
        print("Inspect their prompts and add a regex to CATEGORY_RULES if needed.")


if __name__ == "__main__":
    main()
