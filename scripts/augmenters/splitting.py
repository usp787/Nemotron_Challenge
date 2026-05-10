"""Splitting augmenter: split a single bracket into individually-bracketed symbols.

Inverse of concatenation. Pattern:

    Input:  【]}@]】
    Output: 【]】【}】【@】【]】

Pure RNG over 28 ASCII punctuation symbols. Ported from
nemotron-tonghuikang-source/augmenters/splitting.py with no logic changes.
"""
from __future__ import annotations

import hashlib
import random

SYMBOLS = list('!"#$%&\'()*+-./:;<>?@[\\]^`{|}')

N_PROBLEMS = 1500
LINES_PER_PROBLEM = 100
DEMO_LINES = 3

_LBR = "【"
_RBR = "】"


def _box_individual(chars: list[str]) -> str:
    return "".join(f"{_LBR}{c}{_RBR}" for c in chars)


def _box_merged(chars: list[str]) -> str:
    return f"{_LBR}{''.join(chars)}{_RBR}"


def _random_symbols(rng: random.Random) -> list[str]:
    length = rng.randint(2, 8)
    return [rng.choice(SYMBOLS) for _ in range(length)]


def _pair(chars: list[str]) -> tuple[str, str]:
    # Note: opposite direction from concatenation.
    return _box_merged(chars), _box_individual(chars)


def generate(n_problems: int = N_PROBLEMS, seed: int = 77) -> list[dict[str, str]]:
    """Generate splitting problems. THK uses seed=77 for this augmenter."""
    rng = random.Random(seed)
    problems: list[dict[str, str]] = []

    for i in range(n_problems):
        demo_chars = [_random_symbols(rng) for _ in range(DEMO_LINES)]
        demo_pairs = [_pair(c) for c in demo_chars]

        sample_input_lines = [
            f"{j:02d} {inp}" for j, (inp, _) in enumerate(demo_pairs)
        ]
        sample_output_lines = [
            f"{j:02d} {inp} -> {out}" for j, (inp, out) in enumerate(demo_pairs)
        ]

        test_inputs: list[str] = []
        test_answers: list[str] = []
        for row_num in range(LINES_PER_PROBLEM):
            inp, out = _pair(_random_symbols(rng))
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
        pid = hashlib.sha256(f"splitting_{i}".encode()).hexdigest()[:8]
        problems.append({
            "id": pid,
            "prompt": prompt,
            "completion": completion,
            "category": "splitting",
        })

    return problems
