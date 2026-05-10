"""Lstrip augmenter: strip a single leading space from a bracketed symbol string.

Pattern:

    Input:  【   $%^】
    Output: 【$%^】

Pure RNG. THK generates 300 problems with seed=91. Note the input has
exactly one leading space (the demo "   $%^" rendering in the docstring is
incidental whitespace — the actual augmenter inserts one " " before the
random symbols).

Ported from nemotron-tonghuikang-source/augmenters/lstrip.py.
"""
from __future__ import annotations

import hashlib
import random

SYMBOLS = list('!"#$%&\'()*+-./:;<>?@[\\]^`{|}')

N_PROBLEMS = 300
LINES_PER_PROBLEM = 100
DEMO_LINES = 3

_LBR = "【"
_RBR = "】"


def _box(s: str) -> str:
    return f"{_LBR}{s}{_RBR}"


def _random_entry(rng: random.Random) -> tuple[str, str]:
    if rng.random() < 0.5:
        length = 5
    else:
        length = rng.randint(1, 10)
    symbols = "".join(rng.choice(SYMBOLS) for _ in range(length))
    return _box(" " + symbols), _box(symbols)


def generate(n_problems: int = N_PROBLEMS, seed: int = 91) -> list[dict[str, str]]:
    rng = random.Random(seed)
    problems: list[dict[str, str]] = []

    for i in range(n_problems):
        demo_pairs = [_random_entry(rng) for _ in range(DEMO_LINES)]
        sample_input_lines = [
            f"{j:02d} {inp}" for j, (inp, _) in enumerate(demo_pairs)
        ]
        sample_output_lines = [
            f"{j:02d} {inp} -> {out}" for j, (inp, out) in enumerate(demo_pairs)
        ]

        test_inputs: list[str] = []
        test_answers: list[str] = []
        for row_num in range(LINES_PER_PROBLEM):
            inp, out = _random_entry(rng)
            test_inputs.append(f"{row_num:02d} {inp}")
            test_answers.append(f"{row_num:02d} {inp} -> {out}")

        prompt = (
            "In Alice's Wonderland, secret processing rules are used on text.\n\n"
            "This is a sample input.\n"
            + "\n".join(sample_input_lines)
            + "\n\n"
            + "This is a sample output.\n"
            + "\n".join(sample_output_lines)
            + "\n\n"
            + "This is your input.\n"
            + "\n".join(test_inputs)
        )
        completion = "\n".join(test_answers)
        pid = hashlib.sha256(f"lstrip_{i}".encode()).hexdigest()[:8]
        problems.append({
            "id": pid,
            "prompt": prompt,
            "completion": completion,
            "category": "lstrip",
        })

    return problems
