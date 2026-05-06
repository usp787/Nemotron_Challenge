"""Reasoning generator for the ``gravity`` category.

Task: 3-5 examples of ``For t = T s, distance = D m``. The prompt
itself states the formula d = 0.5 * g * t^2, so g is fully
identified from any single example; we average across all supplied
examples to absorb the per-example rounding noise (ground-truth
answers are quoted to 2 decimal places).

Trace shape: per-example g_i = 2*d/t^2, average, apply to the target.
"""
from __future__ import annotations

import re
from typing import Optional

from .store_types import Problem


_OBS_RE = re.compile(
    r"\s*For\s+t\s*=\s*([\-\d\.]+)\s*s,\s*distance\s*=\s*([\-\d\.]+)\s*m"
)
_TARGET_RE = re.compile(
    r"Now, determine the falling distance for t\s*=\s*([\-\d\.]+)\s*s"
)


def reasoning_gravity(problem: Problem) -> Optional[str]:
    pairs: list[tuple[float, float]] = []
    for line in problem.prompt.splitlines():
        m = _OBS_RE.match(line)
        if m:
            pairs.append((float(m.group(1)), float(m.group(2))))
    qm = _TARGET_RE.search(problem.prompt)
    if not qm or len(pairs) < 1:
        return None
    t_target = float(qm.group(1))

    gs = [2 * d / (t * t) for t, d in pairs if t != 0]
    if not gs:
        return None
    g = sum(gs) / len(gs)
    answer = 0.5 * g * t_target * t_target
    rounded = round(answer, 2)

    lines: list[str] = []
    lines.append(
        "The prompt gives d = 0.5 * g * t^2. Solve each example for g, "
        "then average to cancel the per-row rounding."
    )
    lines.append("")
    lines.append("Per-example g_i = 2 * d / t^2:")
    for t, d in pairs:
        gi = 2 * d / (t * t)
        lines.append(f"  t = {t}, d = {d}: g = 2 * {d} / {t}^2 = {gi:.6f}")
    lines.append("")
    sum_str = " + ".join(f"{gi:.6f}" for gi in gs)
    lines.append(f"g = ({sum_str}) / {len(gs)} = {g:.6f}")
    lines.append("")
    lines.append(f"Apply to t = {t_target}:")
    lines.append(f"  d = 0.5 * {g:.6f} * {t_target}^2")
    lines.append(f"  d = 0.5 * {g:.6f} * {t_target * t_target:.6f}")
    lines.append(f"  d = {answer:.6f}")
    lines.append("")
    lines.append(f"Round to 2 decimal places: {rounded:.2f}")
    lines.append("")
    lines.append(f"The answer in \\boxed{{}} is \\boxed{{{rounded:.2f}}}")
    return "\n".join(lines)
