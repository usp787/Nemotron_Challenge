"""Reasoning generator for the ``numeral`` category.

Task: given a few examples of integer -> Roman numeral, write the
specified integer in Roman. Per the train.csv audit:
  - 1576 problems, all in the same direction (integer -> Roman)
  - integer range: 1..100
  - question phrasing is fixed: "Now, write the number N in the
    Wonderland numeral system."

We emit a deterministic greedy-decomposition trace where every
decision is made explicit (one yes/no comparison per step). At greedy
temp=0 every token's argmax is unambiguous, which is the property
that drives the 100% target solve rate.
"""
from __future__ import annotations

import re
from typing import Optional

from .store_types import Problem


# All Roman value entries the model must consider, descending. We
# emit *every* comparison at every step (not just the matching one),
# so the trace structure is invariant per step -- the model never
# has to learn "which comparisons to skip."
ROMAN_VALUES: list[tuple[str, int]] = [
    ("C", 100),
    ("XC", 90),
    ("L", 50),
    ("XL", 40),
    ("X", 10),
    ("IX", 9),
    ("V", 5),
    ("IV", 4),
    ("I", 1),
]


def reasoning_numeral(problem: Problem) -> Optional[str]:
    m = re.search(r"write the number\s*(-?\d+)", problem.question)
    if not m:
        return None
    n = int(m.group(1))
    if n < 1 or n > 100:
        return None

    lines: list[str] = []
    lines.append(f"We need to write the number {n} as a Wonderland numeral.")
    lines.append("")
    lines.append("Wonderland symbols and values, descending:")
    for sym, val in ROMAN_VALUES:
        lines.append(f"{sym} = {val}")
    lines.append("")
    lines.append(
        "Apply greedy subtraction: at each step, take the largest symbol "
        "whose value fits in n."
    )
    lines.append("")

    symbols: list[str] = []
    cur = n
    while cur > 0:
        lines.append(f"n = {cur}")
        for sym, val in ROMAN_VALUES:
            if cur >= val:
                lines.append(f"{cur} >= {val}? yes, take {sym}, n = {cur - val}")
                symbols.append(sym)
                cur -= val
                break
            else:
                lines.append(f"{cur} >= {val}? no")
        lines.append("")

    lines.append("n = 0, done.")
    lines.append("")
    lines.append(f"Symbols taken in order: {' '.join(symbols)}")
    result = "".join(symbols)
    lines.append(f"Concatenated: {result}")
    lines.append("")
    lines.append(f"The answer in \\boxed{{}} is \\boxed{{{result}}}")

    return "\n".join(lines)
