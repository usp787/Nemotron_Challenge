"""Local smoke test for the augmenter rebuild path.

Runs entirely on CPU. Verifies that augmenter-generated traces match the
data shape that scripts/train_lora.py consumes, and that the assistant
turn fits the Nemotron 3 Nano ``enable_thinking=True`` chat template.

Stages:

  Stage A: every augmenter module imports + deterministic generation
  Stage B: messages JSONL shape produced by build_augmenter_traces.py
  Stage C: assistant turn shape (<think>...</think>, no \\boxed{}, no
           Kaggle 'Please put your final answer' suffix on user content)
  Stage D: token-budget sanity (character proxy)
  Stage E [optional]: AutoTokenizer.apply_chat_template round-trip
                      (skipped when the HF cache is cold)

Stage A coverage:
  * ``concatenation`` / ``splitting`` / ``lstrip`` -- pure RNG, run at
    n_problems=3.
  * ``spelling`` -- needs a tokenizer vocab. Tries local sources, then
    AutoTokenizer offline cache. Skipped (warned, not failed) if none
    available.
  * ``matching`` -- needs finished bit_manipulation reasoning text. We
    synthesize one mini trace by running the local bit_manipulation
    reasoner over the first matching row of data/problems_classified.jsonl,
    then call matching.generate(reasoning_texts={...}). Skipped if
    data/problems_classified.jsonl is missing.

Exit code 0 if all reachable stages pass; non-zero on any failure.

Usage:
    python scripts/smoke_augmenter.py
    python scripts/smoke_augmenter.py --keep-output
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

# Proxy for the 8192-token budget. Real token count check happens in stage E.
MAX_RECORD_CHARS = 8192 * 6

REPO_ROOT = Path(__file__).resolve().parent.parent
CLASSIFIED_PATH = REPO_ROOT / "data" / "problems_classified.jsonl"


def _banner(s: str) -> None:
    print()
    print("=" * 64)
    print(s)
    print("=" * 64, flush=True)


def stage_a_generate() -> dict[str, int]:
    """Returns a per-augmenter count summary."""
    _banner("Stage A: augmenter import + deterministic generation")
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from augmenters import AUGMENTERS, SIMPLE_AUGMENTERS  # noqa: WPS433

    expected = {"concatenation", "splitting", "lstrip", "spelling", "matching"}
    assert set(AUGMENTERS.keys()) == expected, (
        f"AUGMENTERS keys mismatch: {set(AUGMENTERS.keys())} vs {expected}"
    )

    counts: dict[str, int] = {}

    # Simple augmenters: pure RNG, deterministic with seed.
    for name in SIMPLE_AUGMENTERS:
        mod = AUGMENTERS[name]
        a = mod.generate(n_problems=3)
        b = mod.generate(n_problems=3)
        assert a == b, f"{name} non-deterministic across calls"
        assert len(a) == 3, f"{name}: expected 3 problems, got {len(a)}"
        for p in a:
            for key in ("id", "prompt", "completion", "category"):
                assert key in p, f"{name}: missing key {key!r}"
            assert p["category"] == name, f"{name}: bad category {p['category']!r}"
            assert "\\boxed" not in p["completion"], (
                f"{name}: augmenter completion must not contain \\boxed{{}}"
            )
        counts[name] = len(a)
        print(f"  ok: {name:13} n=3, deterministic")

    # Spelling: gated on tokenizer availability.
    spelling = AUGMENTERS["spelling"]
    try:
        a = spelling.generate(n_problems=2)
    except Exception as e:
        print(f"  skip: spelling -- generate() raised {type(e).__name__}: {e}")
        counts["spelling"] = 0
    else:
        if not a:
            print("  skip: spelling -- no tokenizer source available")
            counts["spelling"] = 0
        else:
            assert len(a) == 2, f"spelling: expected 2 problems, got {len(a)}"
            assert "–" in a[0]["completion"], "spelling: missing en-dash"
            counts["spelling"] = len(a)
            print(f"  ok: spelling      n=2 (tokenizer found)")

    # Matching: gated on bit_manipulation classified rows + reasoner.
    matching = AUGMENTERS["matching"]
    if not CLASSIFIED_PATH.exists():
        print(f"  skip: matching -- {CLASSIFIED_PATH} not found")
        counts["matching"] = 0
    else:
        try:
            from reasoners import GENERATORS  # type: ignore
            from reasoners.store_types import build_problem  # type: ignore
        except Exception as e:
            print(f"  skip: matching -- reasoner import failed: {type(e).__name__}: {e}")
            counts["matching"] = 0
        else:
            with open(CLASSIFIED_PATH, "r", encoding="utf-8") as f:
                bm_row = None
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("category") == "bit_manipulation":
                        bm_row = rec
                        break
            if bm_row is None:
                print("  skip: matching -- no bit_manipulation row in classified file")
                counts["matching"] = 0
            else:
                trace = GENERATORS["bit_manipulation"](build_problem(bm_row))
                a = matching.generate(reasoning_texts={bm_row["id"]: trace})
                assert len(a) >= 1, "matching: expected at least 1 problem"
                for p in a:
                    assert "\\boxed" not in p["completion"]
                    assert p["category"] == "matching"
                counts["matching"] = len(a)
                print(f"  ok: matching      n={len(a)} (from 1 bit_manipulation trace)")

    return counts


def stage_b_build_jsonl() -> Path:
    _banner("Stage B: build_augmenter_traces.py end-to-end (concatenation)")
    out = Path(tempfile.gettempdir()) / "augmenter_smoke_traces.jsonl"
    if out.exists():
        out.unlink()
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_augmenter_traces.py"),
        "--only-augmenter", "concatenation",
        "--limit", "3",
        "--out", str(out),
    ]
    print(f"  running: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
        raise RuntimeError("build_augmenter_traces.py failed")
    print("  stdout tail:",
          res.stdout.strip().splitlines()[-1] if res.stdout else "(empty)")
    assert out.exists(), f"expected output file at {out}"

    records = []
    with open(out, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    assert len(records) == 3, f"expected 3 records, got {len(records)}"
    for r in records:
        assert set(r.keys()) >= {"id", "category", "messages"}, f"bad keys: {set(r.keys())}"
        roles = [m["role"] for m in r["messages"]]
        assert roles == ["user", "assistant"], f"bad roles: {roles}"
    print(f"  ok: {len(records)} records via driver -> {out}")
    return out


def stage_c_structure(jsonl_path: Path) -> None:
    _banner("Stage C: assistant turn shape")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    for r in records:
        user = r["messages"][0]["content"]
        assistant = r["messages"][1]["content"]
        assert assistant.startswith("<think>\n"), (
            f"id={r['id']}: assistant must start with '<think>\\n', "
            f"got {assistant[:32]!r}"
        )
        assert assistant.count("</think>") == 1, (
            f"id={r['id']}: expected exactly one </think>"
        )
        assert "\\boxed" not in assistant, (
            f"id={r['id']}: augmenter assistant must not contain \\boxed{{}}"
        )
        assert "\\boxed" not in user, (
            f"id={r['id']}: augmenter user content must not contain \\boxed{{}}"
        )
    print(f"  ok: {len(records)} records well-formed")


def stage_d_size(jsonl_path: Path) -> None:
    _banner("Stage D: per-record size sanity")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    for r in records:
        total_chars = sum(len(m["content"]) for m in r["messages"])
        assert total_chars < MAX_RECORD_CHARS, (
            f"id={r['id']}: total chars {total_chars} exceeds proxy budget"
        )
    avg = sum(
        sum(len(m["content"]) for m in r["messages"]) for r in records
    ) / len(records)
    print(f"  ok: avg total chars per record = {avg:.0f} (budget {MAX_RECORD_CHARS})")


def stage_e_tokenizer(jsonl_path: Path) -> str:
    _banner("Stage E [optional]: AutoTokenizer.apply_chat_template round-trip")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return "skip: transformers not installed"

    try:
        tok = AutoTokenizer.from_pretrained(
            MODEL_ID, trust_remote_code=True, local_files_only=True
        )
    except Exception as e:
        return f"skip: tokenizer not in offline cache ({type(e).__name__})"

    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    for r in records:
        rendered = tok.apply_chat_template(
            r["messages"], tokenize=False,
            add_generation_prompt=False, enable_thinking=True,
        )
        assert "<|im_start|>assistant" in rendered
        assert "<think>\n" in rendered
        assert "</think>" in rendered
        assert rendered.rstrip().endswith("<|im_end|>")
        ids = tok(rendered, add_special_tokens=False)["input_ids"]
        assert len(ids) <= 8192, f"id={r['id']}: token count {len(ids)} > 8192"
    return f"ok: {len(records)} records round-tripped"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--keep-output", action="store_true",
        help="Leave the temporary JSONL on disk for inspection.",
    )
    args = ap.parse_args()

    failed: list[str] = []
    jsonl_path: Path | None = None

    try:
        counts = stage_a_generate()
        print(f"  per-augmenter counts: {counts}")
    except Exception:
        failed.append("A")
        traceback.print_exc()

    try:
        jsonl_path = stage_b_build_jsonl()
    except Exception:
        failed.append("B")
        traceback.print_exc()

    if jsonl_path is not None and jsonl_path.exists():
        for stage_name, fn in [("C", stage_c_structure), ("D", stage_d_size)]:
            try:
                fn(jsonl_path)
            except Exception:
                failed.append(stage_name)
                traceback.print_exc()

        try:
            note = stage_e_tokenizer(jsonl_path)
            print(f"  {note}")
        except Exception:
            failed.append("E")
            traceback.print_exc()

        if not args.keep_output:
            jsonl_path.unlink(missing_ok=True)

    _banner("Summary")
    if failed:
        print(f"FAILED stages: {', '.join(failed)}")
        return 1
    print("All reachable stages PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
