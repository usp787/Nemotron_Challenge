"""Per-category reasoning generators.

Each generator has the signature
    (problem: Problem) -> Optional[str]
and returns a deterministic CoT trace ending with ``\\boxed{<answer>}``,
or ``None`` if the rule isn't recognized for that problem.

Adding a category: write ``reasoners/<name>.py`` with a
``reasoning_<name>`` function, then register it in GENERATORS below.
``build_sft_traces.py`` will pick it up automatically.
"""
from .numeral import reasoning_numeral
from .store_types import Example, Problem, build_problem


GENERATORS = {
    "numeral": reasoning_numeral,
}

__all__ = ["GENERATORS", "Example", "Problem", "build_problem"]
