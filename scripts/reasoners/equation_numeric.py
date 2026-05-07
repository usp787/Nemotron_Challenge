"""Reasoning generator for the ``equation_numeric`` category (deduce mode).

Each problem gives 3-5 examples of ``a OP b = c`` and asks for the
result of a new ``a' OP' b'``. The transformation rule is per-problem
and not bounded -- this deduce-mode solver only handles the *simple*
cases where one of a small fixed catalog of rules fits all examples
verbatim. Anything ambiguous or unmatched returns ``None`` so
``build_sft_traces.py`` drops the problem rather than emitting a guess.

Operand format (audited 2026-05-07): the LHS is always 5 characters --
two operand chars, one operator char, two operand chars -- both for the
numeric sub-type (e.g. ``34/44``) and the symbolic sub-type (e.g.
``%|*"|``). We split on byte index and treat the operator as a third
input to each candidate rule (so rules can be operator-agnostic OR
operator-parameterized).

The yield will be much lower than the algo categories (cipher landed at
~38% even with a perfect substitution solver). What we recover here is
just the deducible slice; the inductive ``_guess`` slice is left to the
base model.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from .store_types import Problem


_QUESTION_RE = re.compile(
    r"Now, determine the result for:\s*(.+?)\s*$", re.MULTILINE
)


def _parse_lhs(lhs: str) -> Optional[tuple[str, str, str]]:
    if len(lhs) != 5:
        return None
    return lhs[:2], lhs[2], lhs[3:]


def _parse_examples(prompt: str) -> list[tuple[str, str, str, str]]:
    examples: list[tuple[str, str, str, str]] = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if " = " not in stripped or stripped.startswith("Now"):
            continue
        lhs, _, rhs = stripped.partition(" = ")
        parts = _parse_lhs(lhs.strip())
        if parts is None:
            continue
        a, op, b = parts
        examples.append((a, op, b, rhs.strip()))
    return examples


def _is_numeric(examples: list[tuple[str, str, str, str]]) -> bool:
    return all(
        a.isdigit() and b.isdigit() and all(c.isdigit() for c in result)
        for a, _, b, result in examples
    )


def _op_arith(a: str, op: str, b: str) -> Optional[str]:
    try:
        ai, bi = int(a), int(b)
    except ValueError:
        return None
    if op == "+":
        return str(ai + bi)
    if op == "-":
        return str(ai - bi)
    if op == "*":
        return str(ai * bi)
    if op == "/" and bi != 0:
        return str(ai // bi)
    return None


# Each rule: (name, fn). fn(a, op, b) returns the predicted result string,
# or None if the rule cannot apply (e.g. divide-by-zero).
Rule = tuple[str, Callable[[str, str, str], Optional[str]]]

_NUMERIC_RULES: list[Rule] = [
    ("a + b (real arithmetic)", lambda a, op, b: str(int(a) + int(b))),
    ("a - b (real arithmetic)", lambda a, op, b: str(int(a) - int(b))),
    ("b - a (real arithmetic)", lambda a, op, b: str(int(b) - int(a))),
    ("|a - b|", lambda a, op, b: str(abs(int(a) - int(b)))),
    ("a * b (real arithmetic)", lambda a, op, b: str(int(a) * int(b))),
    ("a XOR b (numeric)", lambda a, op, b: str(int(a) ^ int(b))),
    ("a OR b (numeric)", lambda a, op, b: str(int(a) | int(b))),
    ("a AND b (numeric)", lambda a, op, b: str(int(a) & int(b))),
    ("(a + b) reversed", lambda a, op, b: str(int(a) + int(b))[::-1]),
    ("(a * b) reversed", lambda a, op, b: str(int(a) * int(b))[::-1]),
    ("concat(a, b)", lambda a, op, b: a + b),
    ("concat(b, a)", lambda a, op, b: b + a),
    (
        "per-digit (a + b) mod 10",
        lambda a, op, b: "".join(str((int(x) + int(y)) % 10) for x, y in zip(a, b)),
    ),
    (
        "per-digit |a - b|",
        lambda a, op, b: "".join(str(abs(int(x) - int(y))) for x, y in zip(a, b)),
    ),
    (
        "per-digit max(a, b)",
        lambda a, op, b: "".join(str(max(int(x), int(y))) for x, y in zip(a, b)),
    ),
    (
        "per-digit min(a, b)",
        lambda a, op, b: "".join(str(min(int(x), int(y))) for x, y in zip(a, b)),
    ),
    (
        "per-digit a XOR b",
        lambda a, op, b: "".join(str(int(x) ^ int(y)) for x, y in zip(a, b)),
    ),
    (
        "per-digit a + b (with carry, drop carry)",
        lambda a, op, b: str(int(a[0]) + int(b[0])) + str(int(a[1]) + int(b[1])),
    ),
    ("op-arith (operator selects + - * //)", _op_arith),
]

_SYMBOLIC_RULES: list[Rule] = [
    ("drop operator from LHS (a + b)", lambda a, op, b: a + b),
    ("reverse LHS", lambda a, op, b: (a + op + b)[::-1]),
    ("a", lambda a, op, b: a),
    ("b", lambda a, op, b: b),
    ("operator + a", lambda a, op, b: op + a),
    ("operator + b", lambda a, op, b: op + b),
    ("a + operator", lambda a, op, b: a + op),
    ("b + operator", lambda a, op, b: b + op),
    ("b + a", lambda a, op, b: b + a),
    ("reverse(a) + reverse(b)", lambda a, op, b: a[::-1] + b[::-1]),
    ("reverse(b) + reverse(a)", lambda a, op, b: b[::-1] + a[::-1]),
    ("reverse(a + b)", lambda a, op, b: (a + b)[::-1]),
    ("a[0] + b[1]", lambda a, op, b: a[0] + b[1]),
    ("a[1] + b[0]", lambda a, op, b: a[1] + b[0]),
    ("a + op + b deduplicated", lambda a, op, b: "".join(dict.fromkeys(a + op + b))),
]


def _safe_eval(rule: Rule, examples: list[tuple[str, str, str, str]]) -> bool:
    name, fn = rule
    try:
        for a, op, b, c in examples:
            res = fn(a, op, b)
            if res is None or res != c:
                return False
        return True
    except Exception:
        return False


def reasoning_equation_numeric(problem: Problem) -> Optional[str]:
    examples = _parse_examples(problem.prompt)
    if len(examples) < 2:
        return None
    qm = _QUESTION_RE.search(problem.prompt)
    if not qm:
        return None
    q_lhs = qm.group(1).strip()
    q_parts = _parse_lhs(q_lhs)
    if q_parts is None:
        return None
    q_a, q_op, q_b = q_parts

    is_numeric = _is_numeric(examples)
    catalog = _NUMERIC_RULES if is_numeric else _SYMBOLIC_RULES

    matches: list[Rule] = []
    rule_results: list[tuple[str, bool]] = []
    for rule in catalog:
        ok = _safe_eval(rule, examples)
        rule_results.append((rule[0], ok))
        if ok:
            matches.append(rule)

    if len(matches) != 1:
        return None

    name, fn = matches[0]
    try:
        answer = fn(q_a, q_op, q_b)
    except Exception:
        return None
    if answer is None:
        return None

    lines: list[str] = []
    sub = "numeric" if is_numeric else "symbolic"
    lines.append(
        f"This is a {sub} equation puzzle: each example follows the form "
        "a OP b = c. Test each candidate rule against every example and "
        "accept it only when it fits all of them."
    )
    lines.append("")
    lines.append("Examples (a, op, b, result):")
    for a, op, b, c in examples:
        lines.append(f"  ({a!r}, {op!r}, {b!r}) -> {c!r}")
    lines.append("")
    lines.append("Candidate rules tested:")
    for rname, ok in rule_results:
        lines.append(f"  {rname}: {'FIT' if ok else 'rejected'}")
    lines.append("")
    lines.append(f"Single fit: {name}")
    lines.append("")
    lines.append(f"Apply to the query ({q_a!r}, {q_op!r}, {q_b!r}):")
    lines.append(f"  result = {answer!r}")
    lines.append("")
    lines.append(f"The answer in \\boxed{{}} is \\boxed{{{answer}}}")
    return "\n".join(lines)
