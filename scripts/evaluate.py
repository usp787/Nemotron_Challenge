"""Compute baseline-report metrics from a predictions JSONL file.

Default mode produces an operational report: total prompts, successes,
failures, latency stats, response length, and a count of common failure
types.

With ``--score``, additionally extracts the final ``\\boxed{...}``
answer from each response and compares it to ``expected_answer`` using
the same rule as the official Kaggle metric (with the binary-strict
addition the author uses in their reasoning.py):

  - If the stored answer is a pure binary string (``[01]+``):
      exact lowercase compare.
  - Else if both sides parse as floats:
      ``math.isclose(rel_tol=1e-2, abs_tol=1e-5)`` (1% tolerance).
  - Else: case-insensitive string compare.

Uses the "last \\boxed{} wins" convention since reasoning traces often
write several boxed expressions during the chain-of-thought and only
the last one is the final answer.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics


_BOXED_RE = re.compile(r"\\boxed\{")
_BOXED_REGEX = re.compile(r"\\boxed\{([^}]*)(?:\}|$)")


def extract_boxed(text: str | None) -> str | None:
    """Return the inner content of the LAST ``\\boxed{...}`` in text.

    Walks braces by hand so nested ``{}`` inside the boxed expression
    parse correctly. Falls back to a regex (matching the Kaggle metric
    exactly) if no balanced ``\\boxed{...}`` is found -- this handles
    truncated traces that end mid-box.
    """
    if not text:
        return None
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    # Truncated \boxed{ at end of text -- mirror the Kaggle regex.
    rx_matches = _BOXED_REGEX.findall(text)
    if rx_matches:
        non_empty = [m.strip() for m in rx_matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return rx_matches[-1].strip()
    return None


def compare_answer(stored: str, predicted: str) -> bool:
    """Mirror Kaggle's official metric (with binary-strict guard)."""
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
    ap.add_argument("--predictions", required=True, help="Path to predictions JSONL")
    ap.add_argument("--score", action="store_true", help="Run \\boxed{} match scoring")
    args = ap.parse_args()

    records: list[dict] = []
    with open(args.predictions, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    total = len(records)
    successes = sum(1 for r in records if r.get("error") is None)
    failures = total - successes

    latencies = [r["latency_sec"] for r in records if isinstance(r.get("latency_sec"), (int, float))]
    resp_lens = [len(r["response"]) for r in records if isinstance(r.get("response"), str)]

    error_counts: dict[str, int] = {}
    for r in records:
        e = r.get("error")
        if not e:
            continue
        kind = e.split(":", 1)[0]
        error_counts[kind] = error_counts.get(kind, 0) + 1

    print(f"Total prompts:           {total}")
    print(f"Successful generations:  {successes}")
    print(f"Failures:                {failures}")
    if latencies:
        print(f"Latency mean (s):        {statistics.mean(latencies):.2f}")
        print(f"Latency median (s):      {statistics.median(latencies):.2f}")
    if resp_lens:
        print(f"Response length mean:    {int(statistics.mean(resp_lens))}")
        print(f"Response length median:  {int(statistics.median(resp_lens))}")
    if error_counts:
        print("Error counts:")
        for kind, n in sorted(error_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {kind}: {n}")

    if not args.score:
        return

    correct = wrong = no_boxed = 0
    for r in records:
        expected = r.get("expected_answer")
        if expected is None:
            continue
        boxed = extract_boxed(r.get("response"))
        if boxed is None:
            no_boxed += 1
            continue
        if compare_answer(str(expected), boxed):
            correct += 1
        else:
            wrong += 1

    scored = correct + wrong + no_boxed
    print()
    print(f"Scored:                  {scored}")
    print(f"Correct:                 {correct}")
    print(f"Wrong:                   {wrong}")
    print(f"No \\boxed{{}}:          {no_boxed}")
    if scored > 0:
        print(f"Accuracy:                {correct / scored:.3f}")


if __name__ == "__main__":
    main()
