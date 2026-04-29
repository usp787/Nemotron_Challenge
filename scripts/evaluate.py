"""Compute baseline-report metrics from a predictions JSONL file.

Default mode produces the operational report described in README
section 12: total prompts, successes, failures, latency stats,
response length, and a count of common failure types.

With ``--score``, additionally extracts the final ``\\boxed{...}``
answer from each response, parses it as an integer, and compares it
to ``expected_answer`` to compute AIME-style accuracy. Uses the
"last \\boxed{} wins" convention since reasoning traces often write
several boxed expressions during the chain-of-thought and only the
last one is the final answer.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics


_BOXED_RE = re.compile(r"\\boxed\{")
_INT_RE = re.compile(r"-?\d+")


def extract_boxed(text: str | None) -> str | None:
    """Return the inner content of the LAST ``\\boxed{...}`` in text.

    Walks braces by hand instead of using a regex so nested ``{}``
    inside the boxed expression (e.g. ``\\boxed{\\frac{1}{2}}``) parse
    correctly. Returns ``None`` if no balanced ``\\boxed{}`` is found.
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
    return None


def parse_int_answer(boxed_content: str | None) -> int | None:
    """Pull the last integer out of boxed content.

    Strips thousands-separator commas and takes the trailing integer so
    answers like ``\\boxed{1,234}``, ``\\boxed{042}``, or
    ``\\boxed{The answer is 42}`` all resolve to the obvious value.
    """
    if boxed_content is None:
        return None
    cleaned = boxed_content.replace(",", "")
    nums = _INT_RE.findall(cleaned)
    if not nums:
        return None
    try:
        return int(nums[-1])
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions JSONL produced by baseline_generate.py",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Also extract \\boxed{...} answers and compute AIME accuracy "
        "against the expected_answer field.",
    )
    args = parser.parse_args()

    total = 0
    successes = 0
    failures = 0
    latencies: list[float] = []
    response_lens: list[int] = []
    errors: dict[str, int] = {}
    records: list[dict] = []

    with open(args.predictions, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)
            total += 1
            if rec.get("error"):
                failures += 1
                key = rec["error"].split(":", 1)[0]
                errors[key] = errors.get(key, 0) + 1
            else:
                successes += 1
                if rec.get("latency_sec") is not None:
                    latencies.append(float(rec["latency_sec"]))
                if rec.get("response"):
                    response_lens.append(len(rec["response"]))

    print(f"Total prompts:           {total}")
    print(f"Successful generations:  {successes}")
    print(f"Failure count:           {failures}")
    if latencies:
        print(f"Average latency (sec):   {statistics.mean(latencies):.3f}")
        print(f"Median latency (sec):    {statistics.median(latencies):.3f}")
    if response_lens:
        print(f"Mean response length:    {statistics.mean(response_lens):.1f} chars")
    if errors:
        print("Common failures:")
        for key, count in sorted(errors.items(), key=lambda kv: -kv[1]):
            print(f"  {key}: {count}")

    if not args.score:
        return

    correct = 0
    wrong = 0
    no_boxed = 0
    no_int = 0
    no_expected = 0
    wrong_rows: list[tuple[str, int | None, int | None]] = []

    for rec in records:
        if rec.get("error") or not rec.get("response"):
            no_boxed += 1
            wrong_rows.append((rec.get("id", "?"), rec.get("expected_answer"), None))
            continue
        expected = rec.get("expected_answer")
        if expected is None:
            no_expected += 1
            continue
        boxed = extract_boxed(rec["response"])
        if boxed is None:
            no_boxed += 1
            wrong_rows.append((rec.get("id", "?"), expected, None))
            continue
        predicted = parse_int_answer(boxed)
        if predicted is None:
            no_int += 1
            wrong_rows.append((rec.get("id", "?"), expected, None))
            continue
        if predicted == int(expected):
            correct += 1
        else:
            wrong += 1
            wrong_rows.append((rec.get("id", "?"), int(expected), predicted))

    scorable = total - no_expected
    print()
    print("Scoring (--score):")
    if no_expected:
        print(f"  Records missing expected_answer:  {no_expected} (not counted)")
    print(f"  Correct:                          {correct} / {scorable}")
    if scorable:
        print(f"  Accuracy:                         {100.0 * correct / scorable:.1f}%")
    print(f"  Wrong (boxed extracted, mismatch): {wrong}")
    print(f"  No \\boxed{{}} found:                {no_boxed}")
    print(f"  Boxed found but non-integer:      {no_int}")
    if wrong_rows:
        print("  Misses (id, expected, predicted):")
        for rid, exp, pred in wrong_rows:
            pred_str = "<no boxed int>" if pred is None else str(pred)
            print(f"    {rid}: expected={exp} predicted={pred_str}")


if __name__ == "__main__":
    main()
