"""Reasoning generator for the ``unit_conversion`` category.

Task: 5 examples of ``X m becomes Y``. Across all training/holdout
problems audited, the transformation is a single multiplicative
factor (y = k * x); per-example ratios agree to 4+ decimal places,
so a simple mean-of-ratios estimate stays well inside the 1e-2
relative tolerance of the eval metric.

Trace shape: list each ratio, average, apply to the target value,
round to 2 decimal places (matching the dataset's answer format).
"""
from __future__ import annotations

import re
from typing import Optional

from .store_types import Problem


_BECOMES_RE = re.compile(r"\s*([\-\d\.]+)\s*\S+\s+becomes\s+([\-\d\.]+)\s*$")
_QUESTION_RE = re.compile(
    r"Now, convert the following measurement:\s*([\-\d\.]+)"
)


def reasoning_unit_conversion(problem: Problem) -> Optional[str]:
    pairs: list[tuple[float, float]] = []
    for line in problem.prompt.splitlines():
        m = _BECOMES_RE.match(line)
        if m:
            pairs.append((float(m.group(1)), float(m.group(2))))
    qm = _QUESTION_RE.search(problem.prompt)
    if not qm or len(pairs) < 2:
        return None
    target = float(qm.group(1))

    ratios = [y / x for x, y in pairs if x != 0]
    if not ratios:
        return None
    k = sum(ratios) / len(ratios)
    answer = k * target
    rounded = round(answer, 2)

    lines: list[str] = []
    lines.append(
        "The Wonderland conversion is linear in the input: y = k * x. "
        "Recover k from each example as y / x, then average."
    )
    lines.append("")
    lines.append("Per-example ratios:")
    for x, y in pairs:
        lines.append(f"  {y} / {x} = {y / x:.6f}")
    lines.append("")
    sum_str = " + ".join(f"{r:.6f}" for r in ratios)
    lines.append(f"k = ({sum_str}) / {len(ratios)} = {k:.6f}")
    lines.append("")
    lines.append(f"Apply to the target measurement {target}:")
    lines.append(f"  y = {k:.6f} * {target} = {answer:.6f}")
    lines.append("")
    lines.append(f"Round to 2 decimal places: {rounded:.2f}")
    lines.append("")
    lines.append(f"The answer in \\boxed{{}} is \\boxed{{{rounded:.2f}}}")
    return "\n".join(lines)
