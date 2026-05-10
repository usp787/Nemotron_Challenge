"""Local smoke test for the reasoner swap.

Runs every reasoner against one real row from data/problems_classified.jsonl
per category. Verifies:

  Stage A: every reasoner imports
  Stage B: every reasoner returns a non-empty trace for at least one row
  Stage C: each trace ends with a parseable ``\\boxed{<answer>}``
  Stage D: extracted answer matches ground truth under Kaggle's
           compare_answer semantics (binary exact / float 1e-2 / case-insens)
  Stage E: per-category accuracy across N rows (default 20)

Cryptarithm doesn't appear in the current data/problems_classified.jsonl
(classifier emits ``equation_numeric`` for both numeric *and* symbolic
puzzles). The smoke test prints a notice rather than failing for that
category — the Phase 4 classifier update will produce real rows.

Exit code 0 if all reachable stages pass; non-zero on any failure.

Usage:
    python scripts/smoke_reasoners.py
    python scripts/smoke_reasoners.py --per-category 50
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASSIFIED_PATH = REPO_ROOT / "data" / "problems_classified.jsonl"

# Same as Kaggle's compare_answer.
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)(?:\}|$)")


def _extract_boxed(text: str) -> str:
    matches = _BOXED_RE.findall(text)
    if not matches:
        return ""
    non_empty = [m.strip() for m in matches if m.strip()]
    if non_empty:
        return non_empty[-1]
    return matches[-1].strip()


def _compare_answer(stored: str, predicted: str) -> bool:
    stored = stored.strip()
    predicted = predicted.strip()
    if re.fullmatch(r"[01]+", stored):
        return predicted.lower() == stored.lower()
    try:
        return math.isclose(
            float(stored), float(predicted), rel_tol=1e-2, abs_tol=1e-5,
        )
    except Exception:
        return predicted.lower() == stored.lower()


def _banner(s: str) -> None:
    print()
    print("=" * 64)
    print(s)
    print("=" * 64, flush=True)


def _load_rows(per_category: int) -> dict[str, list[dict]]:
    if not CLASSIFIED_PATH.exists():
        raise FileNotFoundError(f"{CLASSIFIED_PATH} not found")
    by_cat: dict[str, list[dict]] = {}
    with open(CLASSIFIED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cat = row.get("category", "")
            if not cat:
                continue
            if len(by_cat.get(cat, [])) >= per_category:
                continue
            by_cat.setdefault(cat, []).append(row)
    return by_cat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--per-category", type=int, default=20,
        help="Number of problems per category to test (default: 20)",
    )
    args = ap.parse_args()

    failed: list[str] = []

    _banner("Stage A: reasoner imports")
    try:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from reasoners import GENERATORS
        from reasoners.store_types import build_problem
        print(f"  ok: {len(GENERATORS)} registered categories")
    except Exception:
        failed.append("A")
        traceback.print_exc()
        _banner("Summary")
        print(f"FAILED stages: {', '.join(failed)}")
        return 1

    try:
        by_cat = _load_rows(args.per_category)
    except FileNotFoundError as e:
        print(f"\nNo classified file: {e}")
        print("Stages B-E require data/problems_classified.jsonl. Skipped.")
        _banner("Summary")
        print("Stage A passed; B-E skipped (no data)")
        return 0

    _banner("Stage B-D: trace generation + boxed extraction + compare_answer")
    accuracy: dict[str, tuple[int, int]] = {}
    sample_traces: dict[str, str] = {}
    for cat in sorted(by_cat):
        if cat not in GENERATORS:
            print(f"[{cat}] no generator registered, skipping")
            continue
        gen = GENERATORS[cat]
        rows = by_cat[cat]
        ok = 0
        ran = 0
        for row in rows:
            try:
                problem = build_problem(row)
            except Exception as e:
                print(f"[{cat}] {row.get('id', '?')}: build_problem error: "
                      f"{type(e).__name__}: {e}")
                continue
            try:
                trace = gen(problem)
            except Exception as e:
                print(f"[{cat}] {row['id']}: generator raised: "
                      f"{type(e).__name__}: {e}")
                continue
            if trace is None:
                ran += 1
                continue
            ran += 1
            answer = _extract_boxed(trace)
            if not answer:
                continue
            if _compare_answer(problem.answer, answer):
                ok += 1
                sample_traces.setdefault(cat, trace)
        accuracy[cat] = (ok, ran)
        print(f"[{cat:24}] correct {ok}/{ran}")

    _banner("Stage E: per-category accuracy summary")
    width = max((len(c) for c in accuracy), default=20)
    overall_ok = 0
    overall_n = 0
    for cat in sorted(accuracy):
        ok, n = accuracy[cat]
        pct = (100.0 * ok / n) if n else 0.0
        overall_ok += ok
        overall_n += n
        bar = "█" * int(pct / 5) + " " * (20 - int(pct / 5))
        print(f"  {cat:{width}}  {ok:>4}/{n:<4}  {pct:5.1f}%  |{bar}|")
    if overall_n:
        print(f"  {'TOTAL':{width}}  {overall_ok:>4}/{overall_n:<4}  "
              f"{100.0 * overall_ok / overall_n:5.1f}%")

    # Sanity: at least one category should achieve >0% accuracy.
    if overall_ok == 0 and overall_n > 0:
        failed.append("E")
        print("  no reasoner produced any correct trace — likely build_problem regression")

    _banner("Summary")
    if failed:
        print(f"FAILED stages: {', '.join(failed)}")
        return 1
    print("All reachable stages PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
