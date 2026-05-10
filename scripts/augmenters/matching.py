"""Bit column matching augmenter.

Scrapes the per-operation "Matching output / Left / Right" sub-blocks out of
finished bit_manipulation reasoning traces, and turns each into a standalone
format-following problem (input is the operation bit columns with match
annotations; output is the matching lines + Left chain + Right chain).

In THK's pipeline the input source is ``reasoning/<id>.txt``. Our local
pipeline doesn't dump reasoner text to disk by default, so this port takes
a ``reasoning_texts: dict[problem_id, trace_text]`` mapping. Callers
(e.g. scripts/build_augmenter_traces.py) build that mapping from whatever
source they prefer (in-memory reasoner output, sft_traces.jsonl, or a
THK-style ``reasoning/`` directory).

THK downsamples sparse cases:
  - all-absent matching (no 'match' annotation): keep 1/100
  - both-none chains: keep 1/10
  - <4 matches: keep 1/5
Selection is deterministic via SHA-256 of "<filename>_<section>".

Ported from nemotron-tonghuikang-source/augmenters/matching.py with the
input source generalised.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, Mapping

SECTION_NAMES = [
    "Identity",
    "NOT",
    "Constant",
    "AND",
    "OR",
    "XOR",
    "AND-NOT",
    "OR-NOT",
    "XOR-NOT",
]

# Map section name to Best-line prefix to strip from the trace's Best lines.
_BEST_PREFIX = {
    "Identity": "I",
    "NOT": "NOT",
    "Constant": "C",
    "AND": "AND",
    "OR": "OR",
    "XOR": "XOR",
    "AND-NOT": "AND-NOT",
    "OR-NOT": "OR-NOT",
    "XOR-NOT": "XOR-NOT",
}


def _extract_section_block(
    lines: list[str], start: int, end: int
) -> dict[str, object] | None:
    """Extract Matching output + Left chain + Right chain from lines[start:end]."""
    mo_idx = None
    for i in range(start, end):
        if lines[i].strip() == "Matching output":
            mo_idx = i
            break
    if mo_idx is None:
        return None

    mo_lines: list[str] = []
    for i in range(mo_idx + 1, end):
        stripped = lines[i].strip()
        if stripped == "" or stripped == "Left":
            break
        mo_lines.append(lines[i])

    left_idx = None
    for i in range(mo_idx + 1, end):
        if lines[i].strip() == "Left":
            left_idx = i
            break
    if left_idx is None:
        return None

    left_chain: list[str] = []
    best_left = ""
    for i in range(left_idx + 1, end):
        stripped = lines[i].strip()
        if stripped.startswith("Best:"):
            best_left = lines[i]
            break
        if stripped == "":
            break
        left_chain.append(lines[i])

    right_idx = None
    for i in range(left_idx + 1, end):
        if lines[i].strip() == "Right":
            right_idx = i
            break
    if right_idx is None:
        return None

    right_chain: list[str] = []
    best_right = ""
    for i in range(right_idx + 1, end):
        stripped = lines[i].strip()
        if stripped.startswith("Best:"):
            best_right = lines[i]
            break
        if stripped == "":
            break
        right_chain.append(lines[i])

    return {
        "mo_lines": mo_lines,
        "left_chain": left_chain,
        "best_left": best_left,
        "right_chain": right_chain,
        "best_right": best_right,
    }


def _extract_sections_from_text(
    filename: str, text: str
) -> list[dict[str, object]]:
    """Pull every (operation section, matching/left/right block) pair from one trace."""
    lines = text.split("\n")

    obc_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "Output bit columns (with bitsum as hash)":
            obc_idx = i
            break
    if obc_idx is None:
        return []

    obc_block = lines[obc_idx + 1 : obc_idx + 9]

    selecting_idx = len(lines)
    for i in range(obc_idx, len(lines)):
        if lines[i].strip() == "Selecting":
            selecting_idx = i
            break

    sec_positions: list[tuple[str, int]] = []
    for i in range(obc_idx, selecting_idx):
        stripped = lines[i].strip()
        if stripped in SECTION_NAMES:
            sec_positions.append((stripped, i))

    out: list[dict[str, object]] = []
    for idx, (sec_name, sec_start) in enumerate(sec_positions):
        if idx + 1 < len(sec_positions):
            sec_end = sec_positions[idx + 1][1]
        else:
            sec_end = selecting_idx

        data_lines: list[str] = []
        for i in range(sec_start + 1, sec_end):
            if lines[i].strip() == "Matching output":
                break
            data_lines.append(lines[i])
        while data_lines and data_lines[-1].strip() == "":
            data_lines.pop()

        block = _extract_section_block(lines, sec_start, sec_end)
        if block is None:
            continue

        mo_lines = block["mo_lines"]
        left_chain = block["left_chain"]
        best_left = block["best_left"]
        right_chain = block["right_chain"]
        best_right = block["best_right"]

        all_chain_text = " ".join(left_chain + right_chain)
        has_x = bool(re.search(r"\dx", all_chain_text))
        has_y = bool(re.search(r"\dy", all_chain_text))

        n_matches = sum(1 for l in data_lines if "match" in l)
        all_absent = n_matches == 0
        both_none = left_chain == ["none"] and right_chain == ["none"]

        input_text = "\n".join(obc_block) + "\n\n" + "\n".join(data_lines)

        prefix = _BEST_PREFIX.get(sec_name, sec_name)
        best_left = re.sub(
            rf"^(Best: ){re.escape(prefix)}", r"\1", best_left
        )
        best_right = re.sub(
            rf"^(Best: ){re.escape(prefix)}", r"\1", best_right
        )

        output_parts = (
            mo_lines
            + ["", "Left"]
            + left_chain
            + [best_left]
            + ["", "Right"]
            + right_chain
            + [best_right]
        )
        output_text = "\n".join(output_parts)

        out.append({
            "file": filename,
            "section": sec_name,
            "input": input_text,
            "output": output_text,
            "has_x": has_x,
            "has_y": has_y,
            "all_absent": all_absent,
            "both_none": both_none,
            "few_matches": n_matches < 4,
        })

    return out


def _iter_reasoning_texts(
    reasoning_texts: Mapping[str, str] | None,
    reasoning_dir: Path | str | None,
) -> Iterable[tuple[str, str]]:
    """Resolve (filename, text) pairs from either source."""
    if reasoning_texts is not None:
        for pid, text in reasoning_texts.items():
            yield pid, text
        return
    if reasoning_dir is not None:
        for path in sorted(Path(reasoning_dir).glob("*.txt")):
            yield path.name, path.read_text(encoding="utf-8")


def _extract_all_sections(
    reasoning_texts: Mapping[str, str] | None = None,
    reasoning_dir: Path | str | None = None,
) -> list[dict[str, object]]:
    sections: list[dict[str, object]] = []
    for filename, text in _iter_reasoning_texts(reasoning_texts, reasoning_dir):
        sections.extend(_extract_sections_from_text(filename, text))
    return sections


def generate(
    n_problems: int | None = None,
    reasoning_texts: Mapping[str, str] | None = None,
    reasoning_dir: Path | str | None = None,
) -> list[dict[str, str]]:
    """Build matching problems from finished bit_manipulation reasoner traces.

    Exactly one of ``reasoning_texts`` / ``reasoning_dir`` should be supplied.
    Returns [] if neither yields any bit_manipulation traces.

    ``n_problems`` caps the output (useful for smoke tests). THK doesn't
    cap — they emit one problem per surviving section (after downsampling),
    typically ~4,500 across the corpus.
    """
    if reasoning_texts is None and reasoning_dir is None:
        return []

    sections = _extract_all_sections(reasoning_texts, reasoning_dir)
    if not sections:
        return []

    def _keep(s: dict) -> bool:
        h = int(
            hashlib.sha256(f"{s['file']}_{s['section']}".encode()).hexdigest(),
            16,
        )
        if s["all_absent"]:
            return (h % 100) == 0
        if s["both_none"]:
            return (h % 10) == 0
        if s["few_matches"]:
            return (h % 5) < 1
        return True

    sections = [s for s in sections if _keep(s)]
    if n_problems is not None:
        sections = sections[:n_problems]

    problems: list[dict[str, str]] = []
    for i, item in enumerate(sections):
        prompt = (
            "In Alice's Wonderland, secret processing rules are used on text.\n\n"
            "x: not matched anywhere\n"
            "y: matched but wrong position\n\n"
            + item["input"]
        )
        pid = hashlib.sha256(f"matching_{i}".encode()).hexdigest()[:8]
        problems.append({
            "id": pid,
            "prompt": prompt,
            "completion": item["output"],
            "category": "matching",
        })

    return problems
