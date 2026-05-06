"""Shared types for category reasoners.

Mirrors the shape used in the author's repo (``Problem`` with
``examples`` and ``question``) so per-category solvers can be ported
with minimal edits. Parsing is intentionally generic over the Kaggle
prompt template:

    <header sentence>
    <input1> -> <output1>
    <input2> -> <output2>
    ...
    Now, <verb-phrase> ... <question> ...

Lines containing ``->`` are examples; everything from the first
``Now,`` line onward is the question.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class Example:
    input_value: str
    output_value: str


@dataclass
class Problem:
    id: str
    prompt: str
    answer: str
    category: str
    examples: Sequence[Example] = field(default_factory=tuple)
    question: str = ""


def build_problem(row: dict) -> Problem:
    """Parse a classified row into a Problem with examples + question."""
    examples: list[Example] = []
    question_parts: list[str] = []
    in_question = False
    for line in row["prompt"].splitlines():
        stripped = line.strip()
        if stripped.startswith("Now,"):
            in_question = True
            question_parts.append(stripped)
            continue
        if in_question:
            if stripped:
                question_parts.append(stripped)
            continue
        if " -> " in stripped:
            left, _, right = stripped.partition(" -> ")
            examples.append(Example(left.strip(), right.strip()))
    return Problem(
        id=row["id"],
        prompt=row["prompt"],
        answer=row["answer"],
        category=row["category"],
        examples=tuple(examples),
        question=" ".join(question_parts),
    )
