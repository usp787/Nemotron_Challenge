"""Shared types and long-arithmetic helpers for category reasoners.

Mirrors the shape used in the THK reference repo (``Problem`` with
``examples`` and a *bare-value* ``question``) so the per-category
solvers can be ported with minimal edits.

Prompt parsing
--------------

Kaggle prompts have the structure::

    <header sentence>
    <input1> -> <output1>
    <input2> -> <output2>
    ...
    Now, <verb-phrase> ... <bare-value> ...

``build_problem()`` extracts the example pairs generically (``X -> Y``
lines before the ``Now,`` line) and the ``<bare-value>`` per category:

    bit_manipulation        ->  "00110100"            (8-bit string)
    cipher                  ->  "trb wzrswvog hffk"   (cipher text)
    numeral                 ->  "100"                 (Roman input integer)
    gravity                 ->  "4.41"                (t value)
    unit_conversion         ->  "10.9"                (input measurement)
    equation_numeric*       ->  "85/77"               (5-char a OP b)
    cryptarithm*            ->  "\\(*[#"              (5-char symbolic equation)

The previous build_problem put the entire ``Now, ...`` sentence into
``question``. THK's reasoners expect the bare value, and ours now match.

Long-arithmetic helpers
-----------------------

THK's gravity / unit_conversion / equation_numeric reasoners render step-
by-step long multiplication and long division in their CoT traces. The
helpers were originally in THK's ``store_types.py``; we replicate them
here so the per-category modules can ``from .store_types import ...`` the
same names.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Per-category bare-value extraction
# ---------------------------------------------------------------------------

# Regexes operate on the ``Now, ...`` line only.
_RE_BIT = re.compile(r"output for:\s*([01]+)")
_RE_CIPHER = re.compile(r"decrypt the following text:\s*(.+?)\s*$")
_RE_NUMERAL = re.compile(r"write the number\s+(-?\d+)")
_RE_GRAVITY = re.compile(r"t\s*=\s*([\-\d\.]+)")
_RE_UNIT = re.compile(r"convert the following measurement:\s*([\-\d\.]+)")
_RE_RESULT_FOR = re.compile(r"determine the result for:\s*(.+?)\s*$")

# Example-line regexes. Categories vary in how each input/output pair is rendered:
#   bit_manipulation / cipher / numeral:   "X -> Y"
#   gravity:                                 "For t = X s, distance = Y m"
#   unit_conversion:                         "X m becomes Y"
#   equation_numeric / cryptarithm:          "X = Y"           (NOT the "Now," line)
_RE_EX_GRAVITY = re.compile(
    r"^\s*For\s+t\s*=\s*([\-\d\.]+)\s*s\s*,\s*distance\s*=\s*([\-\d\.]+)\s*m"
)
_RE_EX_UNIT = re.compile(
    r"^\s*([\-\d\.]+)\s*\S+\s+becomes\s+([\-\d\.]+)\s*$"
)
_RE_EX_EQ = re.compile(r"^\s*(\S+)\s*=\s*(\S+)\s*$")


def _find_now_line(prompt: str) -> str:
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("Now,"):
            return stripped
    return ""


def _extract_question(category: str, prompt: str) -> str:
    """Return the bare-value query string for the given category.

    Falls back to "" when the prompt does not match the expected pattern;
    individual reasoners are then responsible for short-circuiting via
    ``return None``.
    """
    now = _find_now_line(prompt)
    if not now:
        return ""

    if category == "bit_manipulation":
        m = _RE_BIT.search(now)
        return m.group(1) if m else ""

    if category == "cipher":
        m = _RE_CIPHER.search(now)
        return m.group(1).strip() if m else ""

    if category == "numeral":
        m = _RE_NUMERAL.search(now)
        return m.group(1) if m else ""

    if category == "gravity":
        m = _RE_GRAVITY.search(now)
        return m.group(1) if m else ""

    if category == "unit_conversion":
        m = _RE_UNIT.search(now)
        return m.group(1) if m else ""

    # equation_numeric* and cryptarithm* share the "determine the result for:"
    # frame. The reasoner itself decides whether the captured group is a
    # numeric equation or a 5-char symbolic equation.
    if category in {
        "equation_numeric",
        "equation_numeric_deduce",
        "equation_numeric_guess",
        "cryptarithm",
        "cryptarithm_deduce",
        "cryptarithm_guess",
    }:
        m = _RE_RESULT_FOR.search(now)
        return m.group(1).strip() if m else ""

    # Unknown category — preserve the full "Now, ..." line as a non-empty
    # default so downstream parsing can still inspect it.
    return now


def _extract_examples(category: str, prompt: str) -> list[Example]:
    """Parse example rows from the prompt. Format varies by category."""
    examples: list[Example] = []
    seen_now = False
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("Now,"):
            seen_now = True
            continue
        if seen_now or not stripped:
            continue

        if category == "gravity":
            m = _RE_EX_GRAVITY.match(stripped)
            if m:
                examples.append(Example(m.group(1), m.group(2)))
            continue

        if category == "unit_conversion":
            m = _RE_EX_UNIT.match(stripped)
            if m:
                examples.append(Example(m.group(1), m.group(2)))
            continue

        if category in {
            "equation_numeric",
            "equation_numeric_deduce",
            "equation_numeric_guess",
            "cryptarithm",
            "cryptarithm_deduce",
            "cryptarithm_guess",
        }:
            m = _RE_EX_EQ.match(stripped)
            if m:
                examples.append(Example(m.group(1), m.group(2)))
            continue

        # Default: "X -> Y" (bit_manipulation, cipher, numeral, and any
        # category we haven't special-cased).
        if " -> " in stripped:
            left, _, right = stripped.partition(" -> ")
            examples.append(Example(left.strip(), right.strip()))

    return examples


def build_problem(row: dict) -> Problem:
    """Parse a classified row into a Problem with examples + bare question."""
    return Problem(
        id=row["id"],
        prompt=row["prompt"],
        answer=row["answer"],
        category=row["category"],
        examples=tuple(_extract_examples(row["category"], row["prompt"])),
        question=_extract_question(row["category"], row["prompt"]),
    )


# ---------------------------------------------------------------------------
# Long-arithmetic helpers
#   Ported from nemotron-tonghuikang-source/reasoners/store_types.py
# ---------------------------------------------------------------------------


def _fmt_int_with_dp(value: int, dp: int) -> str:
    """Format an integer as a decimal string with ``dp`` decimal places."""
    if dp == 0:
        return str(value)
    s = str(value).zfill(dp + 1)
    s = s[: len(s) - dp] + "." + s[len(s) - dp :]
    s = s.lstrip("0") or "0"
    if s.startswith("."):
        s = "0" + s
    return s


def truncate_3dp(s: str) -> str:
    """Truncate (not round) a decimal string to <= 3 decimal places."""
    if "." not in s:
        return s
    integer, frac = s.split(".")
    if len(frac) <= 3:
        return s
    return integer + "." + frac[:3]


def _dp_count(s: str) -> int:
    if "." not in s:
        return 0
    return len(s.split(".")[1])


def pad_dp(s: str, n: int) -> str:
    """Pad a decimal string to exactly ``n`` decimal places (right-zero-fill)."""
    if "." not in s:
        s = s + "."
    integer, frac = s.split(".")
    return integer + "." + frac.ljust(n, "0")


def cast_dp_pair(a: str, b: str) -> tuple[str, str, int, int]:
    """Pad ``a`` and ``b`` to the same number of decimal places.

    Returns ``(a_padded, b_padded, a_target_dp, b_target_dp)`` where the
    two target_dp values are equal (max of the inputs' dp counts).
    """
    da, db = _dp_count(a), _dp_count(b)
    target = max(da, db)
    return pad_dp(a, target), pad_dp(b, target), target, target


def long_multiplication_lines(a_str: str, b_str: str) -> tuple[list[str], str]:
    """Step-by-step decimal multiplication of two decimal strings.

    Decomposes ``b`` into place-value components, multiplies ``a`` by
    each, then shows a left-fold running sum. Returns
    ``(lines, result_str)`` where ``result_str`` is the exact product.
    """
    a_dp = len(a_str.split(".")[1]) if "." in a_str else 0
    b_dp = len(b_str.split(".")[1]) if "." in b_str else 0
    total_dp = a_dp + b_dp

    a_int = int(a_str.replace(".", ""))
    b_int = int(b_str.replace(".", ""))

    lines: list[str] = []

    b_digits_str = str(abs(b_int))
    b_num_digits = len(b_digits_str)

    components: list[tuple[str, int, str]] = []
    for i in range(b_num_digits - 1, -1, -1):
        d = int(b_digits_str[i])
        if d == 0:
            continue
        comp_scaled = d * (10 ** (b_num_digits - 1 - i))
        comp_display = _fmt_int_with_dp(comp_scaled, b_dp)
        if b_dp > 0:
            comp_display = pad_dp(comp_display, b_dp)

        product_int = a_int * comp_scaled
        product_display = _fmt_int_with_dp(product_int, total_dp)
        if total_dp > 0:
            product_display = pad_dp(product_display, total_dp)

        components.append((comp_display, product_int, product_display))

    for comp_display, _, product_display in components:
        lines.append(f"{a_str} * {comp_display} = {product_display}")

    if len(components) >= 2:
        running = components[0][1]
        for i in range(1, len(components)):
            running_display = _fmt_int_with_dp(running, total_dp)
            if total_dp > 0:
                running_display = pad_dp(running_display, total_dp)
            running += components[i][1]
            sum_display = _fmt_int_with_dp(running, total_dp)
            if total_dp > 0:
                sum_display = pad_dp(sum_display, total_dp)
            lines.append(f"{running_display} + {components[i][2]} = {sum_display}")

    total = a_int * b_int
    result_str = _fmt_int_with_dp(total, total_dp)
    return lines, result_str


def long_division_lines(
    numerator_str: str, denominator_str: str, max_decimal_digits: int = 3
) -> tuple[list[str], str]:
    """Step-by-step long division via repeated subtraction.

    Returns ``(lines, result_str)`` where ``result_str`` is the truncated
    quotient (max ``max_decimal_digits`` decimal places).
    """
    n_dp = len(numerator_str.split(".")[1]) if "." in numerator_str else 0
    d_dp = len(denominator_str.split(".")[1]) if "." in denominator_str else 0
    max_dp = max(n_dp, d_dp)

    num = int(round(float(numerator_str) * 10 ** max_dp))
    den = int(round(float(denominator_str) * 10 ** max_dp))

    lines: list[str] = []
    acc = 0
    decimal_digits = 0

    def fmt_acc() -> str:
        if decimal_digits == 0:
            return str(acc)
        s = str(acc).zfill(decimal_digits + 1)
        return s[:-decimal_digits] + "." + s[-decimal_digits:]

    def fmt_scale() -> str:
        if decimal_digits == 0:
            return "1"
        return "0." + "0" * (decimal_digits - 1) + "1"

    def fmt_line(n: int) -> str:
        return f"= {fmt_acc()} + {fmt_scale()} * {n} / {den}"

    lines.append(fmt_line(num))

    while decimal_digits <= max_decimal_digits:
        if num >= den:
            num -= den
            acc += 1
            lines.append(fmt_line(num))
        else:
            decimal_digits += 1
            if decimal_digits > max_decimal_digits:
                break
            num *= 10
            acc *= 10
            lines.append(fmt_line(num))

    if decimal_digits > max_decimal_digits:
        decimal_digits = max_decimal_digits
    return lines, fmt_acc()
