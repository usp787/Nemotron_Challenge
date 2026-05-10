"""Spelling augmenter: break a 3-word line into character-spaced form.

Output format (per row):

    skim cultura olika -> –s–k–i–m–c–u–l–t–u–r–a–o–l–i–k–a–

Spaces are stripped, every char joined by en-dashes, with leading and
trailing en-dashes. Each problem has 100 such rows.

The training tokens themselves are *lowercase, alphabetic, length 2-8*
vocabulary entries from the model's BPE tokenizer. For the bare token
list we read the tokenizer vocab and select entries that look like ASCII
words; for the space-prefixed list we select entries beginning with the
GPT-style space-prefix marker (``\\u0120``, i.e. the "Ġ" glyph).

Tokenizer source priority:
    1. ``--tokenizer-path`` argument if provided
    2. ``data/tokenizer.json`` (drop the THK tokenizer here for self-contained smoke)
    3. ``../nemotron-tonghuikang-source/tokenizer.json`` (sibling source mirror)
    4. ``AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)``
       — used when the HF cache is warm (cluster runtime path)

Each token appears ~4 times across the generated set; n_problems is
derived from the pool size so coverage is balanced.

Ported from nemotron-tonghuikang-source/augmenters/spelling.py.
"""
from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
EN_DASH = "–"
LINES_PER_PROBLEM = 100

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_LOCAL_TOKENIZER = _REPO_ROOT / "data" / "tokenizer.json"
_SIBLING_TOKENIZER = (
    _REPO_ROOT.parent / "nemotron-tonghuikang-source" / "tokenizer.json"
)


def _load_tokens_from_file(path: Path) -> tuple[list[str], list[str]]:
    """Read lowercase 2-8 char alphabetic tokens from a raw tokenizer.json."""
    from tokenizers import Tokenizer  # type: ignore[import-untyped]

    tok = Tokenizer.from_file(str(path))
    vocab = tok.get_vocab()

    bare: list[str] = []
    spaced: list[str] = []
    for token in vocab:
        # GPT-style space-prefix marker is U+0120 (Ġ).
        if token.startswith("Ġ") and len(token) > 1:
            text = token[1:]
            if (
                text.isascii()
                and text.isalpha()
                and text.islower()
                and 2 <= len(text) <= 8
            ):
                spaced.append(text)
        elif (
            token.isascii()
            and token.isalpha()
            and token.islower()
            and 2 <= len(token) <= 8
        ):
            bare.append(token)

    return sorted(bare), sorted(spaced)


def _load_tokens_from_hf() -> tuple[list[str], list[str]]:
    """Fallback: read vocab from a cached AutoTokenizer."""
    from transformers import AutoTokenizer  # type: ignore[import-untyped]

    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, local_files_only=True
    )
    vocab = tok.get_vocab()

    bare: list[str] = []
    spaced: list[str] = []
    for token in vocab:
        if token.startswith("Ġ") and len(token) > 1:
            text = token[1:]
            if (
                text.isascii()
                and text.isalpha()
                and text.islower()
                and 2 <= len(text) <= 8
            ):
                spaced.append(text)
        elif (
            token.isascii()
            and token.isalpha()
            and token.islower()
            and 2 <= len(token) <= 8
        ):
            bare.append(token)
    return sorted(bare), sorted(spaced)


def load_tokens(tokenizer_path: str | Path | None = None) -> tuple[list[str], list[str]]:
    """Return (bare_lowercase_words, space_prefixed_lowercase_words).

    Resolves tokenizer source per the docstring priority.
    """
    if tokenizer_path is not None:
        return _load_tokens_from_file(Path(tokenizer_path))
    if _LOCAL_TOKENIZER.exists():
        return _load_tokens_from_file(_LOCAL_TOKENIZER)
    if _SIBLING_TOKENIZER.exists():
        return _load_tokens_from_file(_SIBLING_TOKENIZER)
    return _load_tokens_from_hf()


def _spell_out(text: str) -> str:
    """Break text into characters, drop spaces, wrap with en-dashes."""
    chars = [c for c in text if c != " "]
    return EN_DASH + EN_DASH.join(chars) + EN_DASH


def generate(
    n_problems: int | None = None,
    seed: int = 42,
    tokenizer_path: str | Path | None = None,
) -> list[dict[str, str]]:
    """Generate spelling problems. THK uses seed=42 for this augmenter.

    If ``n_problems`` is None, size is derived from the vocab pool so each
    token appears ~4 times (THK's default). Pass an int for a smaller pool
    in smoke tests.
    """
    bare_tokens, spaced_tokens = load_tokens(tokenizer_path)
    if not bare_tokens or not spaced_tokens:
        # No vocab available -- the caller (smoke / driver) decides what to do.
        return []

    rng = random.Random(seed)

    bare_shuffled = bare_tokens[:]
    rng.shuffle(bare_shuffled)
    spaced_shuffled = spaced_tokens[:]
    rng.shuffle(spaced_shuffled)

    if n_problems is None:
        n_problems = max(
            math.ceil(len(bare_shuffled) * 4 / LINES_PER_PROBLEM),
            math.ceil(len(spaced_shuffled) * 4 / (LINES_PER_PROBLEM * 2)),
        )

    bare_idx = 0
    spaced_idx = 0
    problems: list[dict[str, str]] = []

    for i in range(n_problems):
        demo_inputs: list[str] = []
        for _ in range(3):
            b = rng.choice(bare_tokens)
            s1, s2 = rng.choice(spaced_tokens), rng.choice(spaced_tokens)
            demo_inputs.append(f"{b} {s1} {s2}")

        sample_input_lines = [f"{j:02d}\n{inp}" for j, inp in enumerate(demo_inputs)]
        sample_output_lines = [
            f"{j:02d}\n{inp} -> {_spell_out(inp)}"
            for j, inp in enumerate(demo_inputs)
        ]

        test_inputs: list[str] = []
        test_answers: list[str] = []
        for row_num in range(LINES_PER_PROBLEM):
            b = bare_shuffled[bare_idx % len(bare_shuffled)]
            bare_idx += 1
            s1 = spaced_shuffled[spaced_idx % len(spaced_shuffled)]
            spaced_idx += 1
            s2 = spaced_shuffled[spaced_idx % len(spaced_shuffled)]
            spaced_idx += 1

            inp = f"{b} {s1} {s2}"
            test_inputs.append(f"{row_num:02d}\n{inp}")
            test_answers.append(f"{row_num:02d}\n{inp} -> {_spell_out(inp)}")

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
        pid = hashlib.sha256(f"spelling_{i}".encode()).hexdigest()[:8]
        problems.append({
            "id": pid,
            "prompt": prompt,
            "completion": completion,
            "category": "spelling",
        })

    return problems
