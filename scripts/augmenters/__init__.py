"""Per-category augmenters.

Augmenters emit short, format-mechanical training examples that teach the
model to follow the structural conventions used by the reasoners (bracketed
symbols, character-spaced spellings, bit-column matchings, etc). Their
completions deliberately do NOT contain ``\\boxed{}`` -- they are pure
scaffolding, never graded by Kaggle.

Each augmenter has the signature

    generate(**kwargs) -> list[dict]

returning records of the form

    {"id": str, "prompt": str, "completion": str, "category": str}

Augmenters split into two groups:

* **Self-contained**: ``concatenation``, ``splitting``, ``lstrip`` produce
  records from pure RNG. The driver calls ``generate(n_problems=...)``.
* **Source-dependent**:
    - ``spelling`` needs a tokenizer vocab (defaults to local discovery,
      see ``spelling.load_tokens``).
    - ``matching`` needs finished bit_manipulation reasoning text; the
      driver passes ``reasoning_texts={pid: trace_text}``.

``scripts/build_augmenter_traces.py`` wraps the output into our standard
``messages`` JSONL shape with assistant content pre-formatted as
``<think>\\n<completion>\\n</think>`` so Nemotron's chat template renders
correctly (see scripts/build_sft_traces.py:restructure_for_thinking).
"""
from . import concatenation, lstrip, matching, spelling, splitting


AUGMENTERS = {
    "concatenation": concatenation,
    "splitting": splitting,
    "lstrip": lstrip,
    "spelling": spelling,
    "matching": matching,
}

# Augmenters that take only ``n_problems`` / ``seed`` (no external dependencies).
# The driver can invoke these without any special argument handling.
SIMPLE_AUGMENTERS = ("concatenation", "splitting", "lstrip")

__all__ = ["AUGMENTERS", "SIMPLE_AUGMENTERS"]
