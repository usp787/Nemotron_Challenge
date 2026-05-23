"""Generate SFT training traces from augmenters.

Augmenters are short, format-mechanical exercises (concatenation, splitting,
lstrip, spelling, matching) that teach the model to follow the structural
conventions used by the reasoners. Their completions have no ``\\boxed{}``.

Output is the same ``{"id", "category", "messages"}`` JSONL shape as
scripts/build_sft_traces.py so the two streams can be concatenated for
training.

Assistant turn shape:

    <think>
    <completion text>
    </think>

This matches Nemotron 3 Nano's ``enable_thinking=True`` chat-template
prefix (which ends in ``<|im_start|>assistant\\n<think>\\n``). The closing
``<|im_end|>`` is added by the chat template at training time. No
``\\boxed{}`` is emitted because augmenters are not graded.

Source plumbing per augmenter:

* ``concatenation`` / ``splitting`` / ``lstrip`` -- self-contained RNG.
* ``spelling`` -- needs a tokenizer vocab. Discovers
  ``data/tokenizer.json`` then ``../nemotron-tonghuikang-source/tokenizer.json``
  then falls back to a cached AutoTokenizer. Override with
  ``--spelling-tokenizer <path>``.
* ``matching`` -- needs finished bit_manipulation reasoning traces.
  Source: ``--matching-source data/sft_traces.jsonl`` (extracts the
  bit_manipulation rows' assistant content) or
  ``--matching-source <dir>`` (THK-style ``reasoning/`` directory of
  ``<id>.txt`` files).

Usage:
    python scripts/build_augmenter_traces.py
    python scripts/build_augmenter_traces.py --only-augmenter concatenation --limit 5
    python scripts/build_augmenter_traces.py --only-augmenter matching --matching-source data/sft_traces.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

# Make `from augmenters import ...` work when run as
# `python scripts/build_augmenter_traces.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from augmenters import AUGMENTERS, SIMPLE_AUGMENTERS  # noqa: E402


def restructure_for_augmenter(completion: str) -> str:
    """Wrap a raw augmenter output as the assistant turn body.

    Mirror of scripts/build_sft_traces.py:restructure_for_thinking, but
    without the ``\\boxed{}`` tail (augmenters have no graded answer).
    The leading ``<think>`` is required: Nemotron's chat template only
    auto-prepends ``<think>`` when the assistant content does not already
    contain ``</think>``.
    """
    return f"<think>\n{completion.rstrip()}\n</think>"


def _strip_thinking_wrapper(content: str) -> str:
    """Recover the raw reasoner trace text from a sft_traces.jsonl assistant turn.

    The assistant content from scripts/build_sft_traces.py:restructure_for_thinking
    looks like::

        <think>
        <reasoning body>
        </think>

        \\boxed{<answer>}

    We want just the ``<reasoning body>``.
    """
    if "<think>\n" in content:
        content = content.split("<think>\n", 1)[1]
    if "\n</think>" in content:
        content = content.split("\n</think>", 1)[0]
    return content.rstrip()


def _load_matching_source_from_sft_traces(path: Path) -> dict[str, str]:
    """Extract bit_manipulation reasoning text from a messages JSONL.

    Records without an ``id`` field are skipped silently -- the current
    scripts/build_sft_traces.py does not emit ids, so this loader is
    forward-compatible with a future update that adds them.
    """
    texts: dict[str, str] = {}
    user_re = re.compile(r"Now, determine the output for:\s*([01]+)")
    fallback_idx = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("category") != "bit_manipulation":
                # Older records lacked category; check the prompt as a fallback.
                user = rec.get("messages", [{}])[0].get("content", "")
                if not user_re.search(user):
                    continue
            assistant = rec["messages"][1]["content"]
            trace = _strip_thinking_wrapper(assistant)
            pid = rec.get("id")
            if not pid:
                pid = f"bm_{fallback_idx:06d}"
                fallback_idx += 1
            texts[pid] = trace
    return texts


def _load_matching_source(arg: str | None) -> dict[str, str] | None:
    if arg is None:
        return None
    path = Path(arg)
    if path.is_dir():
        return {
            p.name: p.read_text(encoding="utf-8") for p in sorted(path.glob("*.txt"))
        }
    if path.is_file():
        return _load_matching_source_from_sft_traces(path)
    raise FileNotFoundError(f"--matching-source not found: {arg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default="data/augmenter_traces.jsonl",
        help="Output JSONL path",
    )
    ap.add_argument(
        "--only-augmenter", default=None,
        help="If set, run only this augmenter (e.g. 'concatenation').",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Per-augmenter cap on generated problems (smoke convenience).",
    )
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for the final shuffle.")
    ap.add_argument(
        "--spelling-tokenizer", default=None,
        help="Path to a tokenizer.json for the spelling augmenter. "
             "Defaults to local discovery (see spelling.load_tokens).",
    )
    ap.add_argument(
        "--matching-source", default=None,
        help="data/sft_traces.jsonl path OR a reasoning/ directory. "
             "Required for the matching augmenter.",
    )
    args = ap.parse_args()

    names = (
        [args.only_augmenter] if args.only_augmenter else sorted(AUGMENTERS.keys())
    )

    matching_source = _load_matching_source(args.matching_source)

    all_records: list[dict] = []
    for name in names:
        mod = AUGMENTERS.get(name)
        if mod is None:
            print(f"[{name}] unknown augmenter, skipping")
            continue

        kwargs: dict = {}
        if args.limit is not None:
            kwargs["n_problems"] = args.limit

        if name == "spelling" and args.spelling_tokenizer is not None:
            kwargs["tokenizer_path"] = args.spelling_tokenizer

        if name == "matching":
            if matching_source is None:
                print(
                    f"[{name}] no --matching-source provided; skipping. "
                    "Run build_sft_traces.py first, then pass "
                    "--matching-source data/sft_traces.jsonl"
                )
                continue
            kwargs["reasoning_texts"] = matching_source

        if name in SIMPLE_AUGMENTERS and args.limit is None:
            # Use module defaults for full runs (so per-category counts match THK).
            kwargs.pop("n_problems", None)

        try:
            problems = mod.generate(**kwargs)
        except Exception as e:
            print(f"[{name}] generator error: {type(e).__name__}: {e}")
            continue

        print(f"[{name}] generated {len(problems)} problems")

        for p in problems:
            content = restructure_for_augmenter(p["completion"])
            all_records.append({
                "id": p["id"],
                "category": p["category"],
                "messages": [
                    {"role": "user", "content": p["prompt"]},
                    {"role": "assistant", "content": content},
                ],
            })

    rng = random.Random(args.seed)
    rng.shuffle(all_records)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_records)} augmenter records to {out}")


if __name__ == "__main__":
    main()
