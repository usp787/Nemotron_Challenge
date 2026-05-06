"""Reasoning generator for the ``cipher`` category.

Task: a per-problem monoalphabetic substitution. Each line of the
form ``CIPHER -> PLAIN`` aligns word-by-word and letter-by-letter,
so we walk the example pairs and accumulate a cipher_letter ->
plain_letter map. When the query contains a letter that no example
covered, the trace is non-deterministic; we return ``None`` so
``build_sft_traces.py`` drops it (rather than emitting a guess that
would fail verification anyway).

Returning ``None`` on a conflict (the same cipher letter mapped to
two different plain letters across examples) is a defensive guard;
the audited training set has no such conflicts, so this only fires
on data corruption.
"""
from __future__ import annotations

import re
from typing import Optional

from .store_types import Problem


_QUESTION_RE = re.compile(
    r"Now, decrypt the following text:\s*(.+?)\s*$", re.MULTILINE
)


def reasoning_cipher(problem: Problem) -> Optional[str]:
    examples: list[tuple[str, str]] = []
    for line in problem.prompt.splitlines():
        if " -> " in line:
            left, _, right = line.partition(" -> ")
            examples.append((left.strip(), right.strip()))
    qm = _QUESTION_RE.search(problem.prompt)
    if not qm or not examples:
        return None
    query = qm.group(1).strip()

    mapping: dict[str, str] = {}
    for cipher, plain in examples:
        cw = cipher.split()
        pw = plain.split()
        if len(cw) != len(pw):
            return None
        for c_word, p_word in zip(cw, pw):
            if len(c_word) != len(p_word):
                return None
            for c_ch, p_ch in zip(c_word, p_word):
                if not (c_ch.isalpha() and p_ch.isalpha()):
                    continue
                c_ch = c_ch.lower()
                p_ch = p_ch.lower()
                if c_ch in mapping and mapping[c_ch] != p_ch:
                    return None
                mapping[c_ch] = p_ch

    for ch in query.lower():
        if ch.isalpha() and ch not in mapping:
            return None

    decoded = "".join(
        mapping[ch.lower()] if ch.isalpha() else ch for ch in query
    )

    lines: list[str] = []
    lines.append(
        "Each example pairs a ciphertext line with its plaintext. "
        "Words and letter positions align, so each cipher letter maps "
        "to exactly one plain letter."
    )
    lines.append("")
    lines.append("Examples:")
    for cipher, plain in examples:
        lines.append(f"  {cipher} -> {plain}")
    lines.append("")
    lines.append("Walk each pair letter-by-letter to build the substitution table:")
    for c in sorted(mapping):
        lines.append(f"  {c} -> {mapping[c]}")
    lines.append("")
    lines.append(f"Apply the table to the query: '{query}'")
    per_char: list[str] = []
    for ch in query:
        if ch.isalpha():
            per_char.append(f"{ch}->{mapping[ch.lower()]}")
        elif ch == " ":
            per_char.append("(space)")
        else:
            per_char.append(f"'{ch}'")
    lines.append("  " + ", ".join(per_char))
    lines.append("")
    lines.append(f"Decoded: {decoded}")
    lines.append("")
    lines.append(f"The answer in \\boxed{{}} is \\boxed{{{decoded}}}")
    return "\n".join(lines)
