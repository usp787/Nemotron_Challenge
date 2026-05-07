"""Reasoning generator for the ``cipher`` category.

Each problem is a per-problem monoalphabetic substitution. We:

1. Walk the example pairs word-by-word and letter-by-letter to
   accumulate the partial substitution table.

2. If every cipher letter in the query is already mapped, decode
   directly (the original "easy" path).

3. Otherwise, run a constraint-propagation pass against the closed
   Wonderland vocabulary (77 words, audited from the full classified
   corpus on 2026-05-07): each query word is matched against every
   vocabulary word of the same length under the partial mapping. If
   exactly one vocabulary word fits, the implied letter assignments
   are added to the mapping (subject to bijectivity), and we iterate.
   This recovers the under-determined problems that the examples
   alone could not solve.

4. After propagation, if the query is still incomplete, the trace is
   abandoned (``return None``) so ``build_sft_traces.py`` drops it
   rather than emitting a guess.
"""
from __future__ import annotations

import re
from typing import Optional

from .store_types import Problem


_QUESTION_RE = re.compile(
    r"Now, decrypt the following text:\s*(.+?)\s*$", re.MULTILINE
)


# Closed Wonderland vocabulary. Generated on 2026-05-07 by scanning every
# cipher problem's plaintext (right-hand side of "->" lines + the answer
# field) in data/problems_classified.jsonl. 77 unique words, lowercased.
_VOCAB: tuple[str, ...] = (
    "above", "alice", "ancient", "around", "beyond", "bird", "book",
    "bright", "castle", "cat", "cave", "chases", "clever", "colorful",
    "creates", "crystal", "curious", "dark", "discovers", "door",
    "dragon", "draws", "dreams", "explores", "follows", "forest",
    "found", "garden", "golden", "hatter", "hidden", "imagines", "in",
    "inside", "island", "key", "king", "knight", "library", "magical",
    "map", "message", "mirror", "mountain", "mouse", "mysterious",
    "near", "ocean", "palace", "potion", "princess", "puzzle", "queen",
    "rabbit", "reads", "school", "secret", "sees", "silver", "story",
    "strange", "student", "studies", "teacher", "the", "through",
    "tower", "treasure", "turtle", "under", "valley", "village",
    "watches", "wise", "wizard", "wonderland", "writes",
)
_VOCAB_BY_LEN: dict[int, tuple[str, ...]] = {}
for _w in _VOCAB:
    _VOCAB_BY_LEN.setdefault(len(_w), [])
    _VOCAB_BY_LEN[len(_w)].append(_w)
_VOCAB_BY_LEN = {k: tuple(v) for k, v in _VOCAB_BY_LEN.items()}


def _build_partial_mapping(
    examples: list[tuple[str, str]],
) -> Optional[dict[str, str]]:
    """Walk word-aligned example pairs and accumulate cipher->plain.

    Returns ``None`` on any structural inconsistency (word-count or
    letter-count mismatch, or two example pairs disagreeing on the same
    cipher letter).
    """
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
    return mapping


def _word_candidates(
    cipher_word: str, mapping: dict[str, str]
) -> list[tuple[str, dict[str, str]]]:
    """Vocabulary words that fit ``cipher_word`` under ``mapping``.

    Returns a list of (vocab_word, implied_new_assignments). Bijectivity
    against the existing mapping is enforced: a cipher letter cannot
    map to a plain letter that another cipher letter already owns.
    """
    cw = cipher_word.lower()
    if not all(c.isalpha() for c in cw):
        return []
    pool = _VOCAB_BY_LEN.get(len(cw), ())
    used_plain = set(mapping.values())

    out: list[tuple[str, dict[str, str]]] = []
    for cand in pool:
        local: dict[str, str] = {}
        ok = True
        local_used: set[str] = set()
        for c_ch, p_ch in zip(cw, cand):
            if c_ch in mapping:
                if mapping[c_ch] != p_ch:
                    ok = False
                    break
            elif c_ch in local:
                if local[c_ch] != p_ch:
                    ok = False
                    break
            else:
                if p_ch in used_plain or p_ch in local_used:
                    ok = False
                    break
                local[c_ch] = p_ch
                local_used.add(p_ch)
        if ok:
            out.append((cand, local))
    return out


def _propagate(
    mapping: dict[str, str],
    query_words: list[str],
    log: list[str],
) -> dict[str, str]:
    """Iterate vocabulary-constraint propagation until fixed point.

    Each round, for every query word with one or more unmapped letters,
    enumerate vocabulary candidates. If exactly one candidate fits, the
    implied letter assignments are committed. Stop when no round commits
    new letters.
    """
    mapping = dict(mapping)
    while True:
        changed = False
        for word in query_words:
            cw = word.lower()
            if all((not c.isalpha()) or c in mapping for c in cw):
                continue
            cands = _word_candidates(cw, mapping)
            if len(cands) == 1:
                vocab_word, new_map = cands[0]
                if new_map:
                    log.append(
                        f"  '{word}' uniquely matches '{vocab_word}'; add "
                        + ", ".join(f"{k}->{v}" for k, v in new_map.items())
                    )
                    mapping.update(new_map)
                    changed = True
            elif len(cands) > 1:
                preview = ", ".join(c for c, _ in cands[:5])
                log.append(
                    f"  '{word}' has {len(cands)} candidates ({preview}"
                    + ("..." if len(cands) > 5 else "")
                    + "); skipping until other words narrow it down."
                )
        if not changed:
            return mapping


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

    mapping = _build_partial_mapping(examples)
    if mapping is None:
        return None

    query_words = [w for w in query.split() if w]
    needs_inference = any(
        ch.isalpha() and ch.lower() not in mapping for w in query_words for ch in w
    )

    propagation_log: list[str] = []
    if needs_inference:
        before = dict(mapping)
        mapping = _propagate(mapping, query_words, propagation_log)
        # If inference resolved nothing useful, propagation_log may still
        # be empty; that's fine -- we just won't include the section.
        if mapping == before:
            propagation_log.append("  (no new letters resolvable from vocabulary)")

    # Final check: every alphabetic letter in the query must be mapped.
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

    if propagation_log:
        lines.append(
            "Some letters in the query are not covered by the examples. "
            "Resolve them by matching each query word against the closed "
            "Wonderland vocabulary under the partial table:"
        )
        lines.extend(propagation_log)
        lines.append("")
        lines.append("Updated substitution table:")
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
