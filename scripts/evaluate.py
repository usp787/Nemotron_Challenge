"""Compute baseline-report metrics from a predictions JSONL file.

Produces the minimum baseline report described in README section 12:
total prompts, successes, failures, latency stats, response length,
and a count of common failure types.
"""
import argparse
import json
import statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions JSONL produced by baseline_generate.py",
    )
    args = parser.parse_args()

    total = 0
    successes = 0
    failures = 0
    latencies: list[float] = []
    response_lens: list[int] = []
    errors: dict[str, int] = {}

    with open(args.predictions, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
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


if __name__ == "__main__":
    main()
