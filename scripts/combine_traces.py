"""Combine reasoner + augmenter trace files into one shuffled JSONL.

Reads ``data/sft_traces.jsonl`` (reasoner output) and
``data/augmenter_traces.jsonl`` (augmenter output) and writes a single
shuffled training file. Records keep their existing ``id`` / ``category``
fields so per-category metrics work downstream.

Optional length filter (``--max-chars N``): drop any record whose
combined user+assistant content exceeds N characters. Defaults to a
*generous* 8192 * 6 = 49,152 character ceiling, which is a safe
proxy for the 8192-token Kaggle / vLLM limit (text is roughly 4 chars
per BPE token in this corpus; 6x gives plenty of slack). The real
token-level filter happens at training time inside scripts/train_lora.py
where the HF tokenizer is available.

Usage:
    python scripts/combine_traces.py
    python scripts/combine_traces.py --out data/sft_combined.jsonl --max-chars 49152
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reasoner", default="data/sft_traces.jsonl")
    ap.add_argument("--augmenter", default="data/augmenter_traces.jsonl")
    ap.add_argument("--out", default="data/sft_combined.jsonl")
    ap.add_argument(
        "--max-chars", type=int, default=8192 * 6,
        help="Drop records whose user+assistant content exceeds this many "
             "chars. Proxy for the 8192-token budget; real token check "
             "happens at training time. Default: 49152.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    reasoner = _load(Path(args.reasoner))
    augmenter = _load(Path(args.augmenter))
    print(f"Reasoner records: {len(reasoner)}")
    print(f"Augmenter records: {len(augmenter)}")

    all_rows = reasoner + augmenter
    print(f"Before length filter: {len(all_rows)}")

    kept: list[dict] = []
    dropped_by_cat: Counter = Counter()
    for r in all_rows:
        n_chars = sum(len(m["content"]) for m in r["messages"])
        if n_chars > args.max_chars:
            dropped_by_cat[r.get("category", "?")] += 1
            continue
        kept.append(r)
    print(f"After length filter (max_chars={args.max_chars}): {len(kept)}")
    if dropped_by_cat:
        print(f"Dropped by category (likely cipher / equation_numeric heavy):")
        for k, v in sorted(dropped_by_cat.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")

    rng = random.Random(args.seed)
    rng.shuffle(kept)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_cat: Counter = Counter()
    for r in kept:
        by_cat[r.get("category", "?")] += 1
    print()
    print(f"Final corpus -> {out}")
    print("Per-category record count:")
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
