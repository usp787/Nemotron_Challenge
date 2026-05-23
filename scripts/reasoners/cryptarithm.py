"""Equation symbolic reasoning generator.

Default behaviour (THK 04-10-04-33 parity) handles concatenation operators
only — forward (LHS + RHS) and reverse (RHS + LHS). Operates directly on
the original symbols without letter assignment.

Extended hypothesis search: when concat doesn't fit, the reasoner now
also tries position-permutation rules — every non-empty subset of input
positions {0,1,2,3,4} with every ordering. Empirically this lifts the
boxed-correctness rate on cryptarithm_deduce from ~8.0% to ~9.4% (the
extra 1.4% picks up problems whose hidden rule is a positional projection
like "keep chars 0,3,4 in that order"). cryptarithm_guess remains close
to 0% solvable by simple rules; see the rebuild diagnostics doc.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations

from .store_types import Problem


@dataclass
class _Ex:
    a: tuple[str, str]
    op: str
    b: tuple[str, str]
    out: str


_ALL_POSITIONS = (0, 1, 2, 3, 4)
# All non-empty position permutations of {0..4}. 5+5*4+5*4*3+5*4*3*2+5! = 325
# orderings. Precomputed at import time so the inner loop stays tight.
_POSITION_PERMS: tuple[tuple[int, ...], ...] = tuple(
    perm
    for r in range(1, 6)
    for combo in combinations(_ALL_POSITIONS, r)
    for perm in permutations(combo)
)


def _apply_perm(inp: str, perm: tuple[int, ...]) -> str:
    """Apply a position permutation to a 5-char cryptarithm input."""
    return "".join(inp[i] for i in perm)


def _perm_label(perm: tuple[int, ...]) -> str:
    """Human-readable rule label, e.g. 'LHS' / 'RHS' / 'drop-op' / 'pos[0,3,4]'."""
    canonical = {
        (0, 1, 3, 4): "drop-operator",
        (3, 4, 0, 1): "reverse halves",
        (0, 1): "LHS",
        (3, 4): "RHS",
        (1, 0): "reversed LHS",
        (4, 3): "reversed RHS",
        (4, 3, 1, 0): "fully reversed (op dropped)",
        (4, 3, 2, 1, 0): "fully reversed",
        (0, 1, 2, 3, 4): "identity",
    }
    if perm in canonical:
        return canonical[perm]
    return "positions[" + ",".join(str(i) for i in perm) + "]"


def _find_perm_rule(exs: list[_Ex]) -> tuple[int, ...] | None:
    """Find a position perm that maps each example's input to its output.

    Returns the first matching perm in canonical search order (LHS-first,
    drop-op, reverse halves, then arbitrary perms). Returns None if no
    single perm fits all examples — including the trivial case where the
    outputs across examples vary in length.
    """
    if not exs:
        return None
    expected_len = len(exs[0].out)
    if any(len(ex.out) != expected_len for ex in exs):
        # Variable-length outputs can't be a single position perm.
        return None
    if expected_len < 1 or expected_len > 5:
        return None
    # Reconstruct the original 5-char inputs from the parsed (a, op, b) shape.
    inputs = [
        ex.a[0] + ex.a[1] + ex.op + ex.b[0] + ex.b[1] for ex in exs
    ]
    for perm in _POSITION_PERMS:
        if len(perm) != expected_len:
            continue
        if all(_apply_perm(inp, perm) == ex.out for inp, ex in zip(inputs, exs)):
            return perm
    return None


def _concat_type(exs: list[_Ex]) -> str | None:
    """Return 'fwd' if A1A2B1B2, 'rev' if B1B2A1A2, else None."""
    if all(ex.out == ex.a[0] + ex.a[1] + ex.b[0] + ex.b[1] for ex in exs):
        return "fwd"
    if all(ex.out == ex.b[0] + ex.b[1] + ex.a[0] + ex.a[1] for ex in exs):
        return "rev"
    return None


def _box(s: str) -> str:
    """Wrap each character in 【】 brackets."""
    return "".join(f"【{c}】" for c in s)


def reasoning_cryptarithm(problem: Problem) -> str | None:
    """Generate reasoning for cryptarithm problems."""

    def quote(s: str) -> str:
        return f"【{s}】"

    exs: list[_Ex] = []
    for ex in problem.examples:
        inp = str(ex.input_value)
        if len(inp) != 5:
            return None
        exs.append(
            _Ex(
                a=(inp[0], inp[1]),
                op=inp[2],
                b=(inp[3], inp[4]),
                out=str(ex.output_value),
            )
        )

    q = str(problem.question)
    if len(q) != 5:
        return None
    q_a = (q[0], q[1])
    q_op = q[2]
    q_b = (q[3], q[4])

    # Group by operator
    by_op: dict[str, list[_Ex]] = {}
    for parsed_ex in exs:
        by_op.setdefault(parsed_ex.op, []).append(parsed_ex)

    # Detect concat types for each operator
    concat_types: dict[str, str] = {}
    for op, op_exs in by_op.items():
        ct = _concat_type(op_exs)
        if ct is not None:
            concat_types[op] = ct

    # Position-permutation fallback. For each operator whose examples don't
    # follow a simple fwd/rev concat, search the position-perm space for a
    # single perm that maps every input to its output. When found, treat it
    # as the rule for that operator. Position perms can produce outputs of
    # any length from 1 to 5, which is the actual format range in the data.
    perm_rules: dict[str, tuple[int, ...]] = {}
    for op, op_exs in by_op.items():
        if op in concat_types:
            continue
        perm = _find_perm_rule(op_exs)
        if perm is not None:
            perm_rules[op] = perm

    # Resolve the question's rule, in priority order:
    #   1. concat rule for q_op
    #   2. position-perm rule for q_op (only when q_op has its own examples)
    #   3. fall back to forward concat (THK default)
    #
    # NOTE: an earlier draft also inherited perm rules cross-operator when
    # q_op was absent from examples. Diagnostics showed this slightly hurt
    # cryptarithm_guess (-4 problems / 164) because the "donor" op's perm
    # rarely transfers to the question op; the resulting wrong answer
    # displaced the lucky fwd-concat default that happened to match. We
    # keep cross-op inheritance OFF and rely on --drop-wrong-hard-categories
    # in build_sft_traces.py to remove the wrong traces instead.
    q_input = q_a[0] + q_a[1] + q_op + q_b[0] + q_b[1]
    q_ct: str | None = None
    q_perm: tuple[int, ...] | None = None
    q_rule_source: str = ""
    if q_op in concat_types:
        q_ct = concat_types[q_op]
        q_rule_source = f"concat({q_ct})"
    elif q_op in perm_rules:
        q_perm = perm_rules[q_op]
        q_rule_source = f"perm {_perm_label(q_perm)}"
    else:
        q_ct = "fwd"
        q_rule_source = "concat(fwd) default"

    if q_perm is not None:
        answer = _apply_perm(q_input, q_perm)
    elif q_ct == "fwd":
        answer = q_a[0] + q_a[1] + q_b[0] + q_b[1]
    else:
        answer = q_b[0] + q_b[1] + q_a[0] + q_a[1]

    # Generate trace
    lines: list[str] = []
    lines.append("We need to infer the transformation rule from the examples.")
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")

    # Show each example with concatenation check
    for ex, ex_parsed in zip(problem.examples, exs):
        orig_inp = str(ex.input_value)
        orig_out = str(ex.output_value)
        lines.append(f"{quote(orig_inp)} = {quote(orig_out)}")
        a0, a1 = quote(ex_parsed.a[0]), quote(ex_parsed.a[1])
        b0, b1 = quote(ex_parsed.b[0]), quote(ex_parsed.b[1])
        op_q = quote(ex_parsed.op)
        out_boxed = _box(orig_out)
        lines.append(f"  input: {a0}{a1}{op_q}{b0}{b1}")
        lines.append(f"  left:{a0}{a1}")
        lines.append(f"  operator: {op_q}")
        lines.append(f"  right:{b0}{b1}")
        lines.append(f"  output: {out_boxed}")

        fwd = ex_parsed.a[0] + ex_parsed.a[1] + ex_parsed.b[0] + ex_parsed.b[1]
        rev = ex_parsed.b[0] + ex_parsed.b[1] + ex_parsed.a[0] + ex_parsed.a[1]
        is_fwd = orig_out == fwd
        is_rev = orig_out == rev

        lines.append(
            f"  concatenation: {_box(fwd)} {'match' if is_fwd else 'mismatch'}"
        )
        lines.append(
            f"  reverse concatenation: {_box(rev)} {'match' if is_rev else 'mismatch'}"
        )

        # Operator line with type
        ct = concat_types.get(ex_parsed.op)
        if ct == "fwd":
            op_type = "concatenation"
        elif ct == "rev":
            op_type = "reverse concatenation"
        else:
            op_type = "unknown"
        lines.append(f"  operator: {quote(ex_parsed.op)}{op_type}")
        lines.append("")

    # Apply to question
    qa0, qa1 = quote(q_a[0]), quote(q_a[1])
    qb0, qb1 = quote(q_b[0]), quote(q_b[1])
    q_orig = str(problem.question)
    lines.append(f"Question{quote(q_orig)}")
    lines.append(f"  input: {qa0}{qa1}{quote(q_op)}{qb0}{qb1}")
    lines.append(f"  left:{qa0}{qa1}")
    lines.append(f"  operator:{quote(q_op)}")
    lines.append(f"  right:{qb0}{qb1}")
    lines.append("")

    lines.append(f"The question operator is {quote(q_op)}. Rule: {q_rule_source}.")

    if q_perm is not None:
        # Position-perm trace: enumerate positions and the chars taken.
        picked = ", ".join(
            f"pos {p} = {quote(q_input[p])}" for p in q_perm
        )
        lines.append(f"  picking ({picked}) -> {_box(answer)}")
    else:
        op_label = "concatenation" if q_ct == "fwd" else "reverse concatenation"
        lines.append(
            f"  {op_label}({qa0}{qa1}, {qb0}{qb1}) = {_box(answer)}"
        )
    lines.append(f"  output: {quote(answer)}-> {quote('{' + answer + '}')}")
    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{answer}}}")
    return "\n".join(lines)
