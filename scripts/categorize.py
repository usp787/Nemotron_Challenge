"""Classify each train.csv row by Kaggle category.

The Kaggle prompts use a deterministic templated header per category.
Output is one JSONL record per row with ``{id, prompt, answer, category}``.

Categories emitted (9, matching THK's split):

    bit_manipulation
    cipher
    cryptarithm_deduce      — symbolic equations, question op IS in examples
    cryptarithm_guess       — symbolic equations, question op is NOT
    equation_numeric_deduce — numeric equations, question op IS in examples
    equation_numeric_guess  — numeric equations, question op is NOT
    gravity
    numeral
    unit_conversion

The Kaggle header "a secret set of transformation rules is applied to
equations" covers both numeric and symbolic puzzles, so after the
header regex matches, ``_refine_equation_header()`` inspects the example
operand/operator characters to split numeric vs symbolic, then checks
whether the question operator appears in any example to split
deduce vs guess.

Run on the local box; output is small and gets committed to git so the
cluster receives it via ``git pull``.

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


# Header-line regex per category. The ``equation_numeric`` header is
# shared with cryptarithm in the Kaggle dataset; rows matching it get
# refined further by ``_refine_equation_header``.
CATEGORY_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("bit_manipulation",   re.compile(r"bit manipulation rule transforms 8-bit binary numbers")),
    ("cipher",             re.compile(r"secret encryption rules are used on text")),
    ("numeral",            re.compile(r"numbers are secretly converted into a different numeral system")),
    ("unit_conversion",    re.compile(r"a secret unit conversion is applied to measurements")),
    ("gravity",            re.compile(r"the gravitational constant has been secretly changed")),
    ("equation_or_crypt",  re.compile(r"a secret set of transformation rules is applied to equations")),
]


_RE_RESULT_FOR = re.compile(r"Now, determine the result for:\s*(.+?)\s*$", re.MULTILINE)


def _parse_eq_examples_and_question(
    prompt: str,
) -> tuple[list[tuple[str, str, str, str]], tuple[str, str, str] | None]:
    """Return (examples=[(a, op, b, out), ...], question=(a, op, b) | None).

    All equation-style LHS values are 5 chars: aaOPbb (two operand chars,
    one operator char, two operand chars). RHS is whatever the rule produced.
    """
    examples: list[tuple[str, str, str, str]] = []
    question: tuple[str, str, str] | None = None

    qm = _RE_RESULT_FOR.search(prompt)
    if qm:
        q_lhs = qm.group(1).strip()
        if len(q_lhs) == 5:
            question = (q_lhs[0:2], q_lhs[2], q_lhs[3:5])

    for line in prompt.splitlines():
        s = line.strip()
        if not s or s.startswith("Now,"):
            continue
        if " = " not in s:
            continue
        lhs, _, rhs = s.partition(" = ")
        if len(lhs) == 5:
            examples.append((lhs[0:2], lhs[2], lhs[3:5], rhs.strip()))

    return examples, question


def _refine_equation_header(prompt: str) -> str:
    """Resolve one of the 4 deduce/guess buckets, or fall back to a default."""
    examples, q = _parse_eq_examples_and_question(prompt)
    if not examples or q is None:
        # Shape we don't understand. Fall back to the legacy combined label
        # so downstream consumers still see something registerable.
        return "equation_numeric"

    # THK's numeric/symbolic split: every operand char AND every RHS char
    # must be a digit for the puzzle to count as "numeric".
    is_numeric = all(
        a.isdigit() and b.isdigit() and result.isdigit()
        for a, _, b, result in examples
    )

    # Deduce/guess split: question operator present in any example?
    q_op = q[1]
    op_in_examples = any(ex[1] == q_op for ex in examples)

    if is_numeric:
        return "equation_numeric_deduce" if op_in_examples else "equation_numeric_guess"
    return "cryptarithm_deduce" if op_in_examples else "cryptarithm_guess"


def classify(prompt: str) -> str:
    for name, pat in CATEGORY_RULES:
        if pat.search(prompt):
            if name == "equation_or_crypt":
                return _refine_equation_header(prompt)
            return name
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="train&test/train.csv")
    ap.add_argument("--out", default="data/problems_classified.jsonl")
    ap.add_argument(
        "--thk-labels", default=None,
        help="Path to THK's problems.jsonl (authoritative deduce/guess split). "
             "When provided, labels for ids present in this file override the "
             "regex-derived label. Our regex split agrees with THK on the 5 "
             "single-header categories but produces a slightly different "
             "equation/cryptarithm split (the heuristic is more nuanced than "
             "'is question op in examples'). Default: "
             "../nemotron-tonghuikang-source/problems.jsonl if it exists.",
    )
    args = ap.parse_args()

    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    counts: Counter[str] = Counter()
    overrides_used = 0

    # Resolve THK label source.
    thk_labels: dict[str, str] = {}
    thk_path = args.thk_labels
    if thk_path is None:
        default = Path(__file__).resolve().parent.parent.parent / "nemotron-tonghuikang-source" / "problems.jsonl"
        if default.exists():
            thk_path = str(default)
    if thk_path:
        with open(thk_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    thk_labels[rec["id"]] = rec["category"]
        print(f"Loaded {len(thk_labels)} THK authoritative labels from {thk_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.train_csv, "r", encoding="utf-8", newline="") as f, \
         open(out_path, "w", encoding="utf-8") as out:
        reader = csv.DictReader(f)
        for row in reader:
            local_cat = classify(row["prompt"])
            cat = thk_labels.get(row["id"], local_cat)
            if cat != local_cat:
                overrides_used += 1
            counts[cat] += 1
            out.write(json.dumps({
                "id": row["id"],
                "prompt": row["prompt"],
                "answer": row["answer"],
                "category": cat,
            }, ensure_ascii=False) + "\n")

    total = sum(counts.values())
    print(f"Classified {total} rows -> {out_path}")
    if thk_labels:
        print(f"THK overrides applied: {overrides_used} (rows where our regex "
              "differed from THK's authoritative deduce/guess split)")
    for cat, n in counts.most_common():
        pct = 100.0 * n / total if total else 0.0
        print(f"  {cat:30} {n:>6} ({pct:.1f}%)")
    if counts.get("unknown", 0):
        print()
        print(f"WARNING: {counts['unknown']} rows did not match any category rule.")
        print("Inspect their prompts and add a regex to CATEGORY_RULES if needed.")


if __name__ == "__main__":
    main()
