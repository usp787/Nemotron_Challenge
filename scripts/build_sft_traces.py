"""Generate SFT training traces and per-category holdout JSONLs.

Reads ``data/problems_classified.jsonl``, dispatches each problem to
the matching reasoner, and writes:

  data/sft_traces.jsonl              (training, ``messages`` format)
  data/<category>_holdout.jsonl     (per-category eval set, one file each)

Each training record is shaped as::

    {"id", "category", "messages": [{"role": "user", ...},
                                    {"role": "assistant", ...}]}

The ``id`` + ``category`` fields are always emitted so downstream consumers
(scripts/build_augmenter_traces.py's matching augmenter, for one) can
filter to bit_manipulation rows without re-parsing.

Two modes for the assistant target:

  default (baseline)
    Wrong-answer traces are dropped. The assistant's ``\\boxed{...}`` is
    rewritten to the *ground truth* answer. Maximises per-trace correctness
    but discards the reasoner's procedural signal on hard puzzles.

  ``--keep-wrong-traces --use-reasoner-boxed``  (matches THK's 04-10-04-33)
    Every reasoner output is kept regardless of correctness. The
    assistant's ``\\boxed{...}`` value is whatever the reasoner itself
    emitted -- often wrong on cryptarithm and equation_numeric_guess, but
    the surrounding procedure is still a valid CoT pattern. THK's
    LB-winning run trains on this superset (~17K records) because the
    procedure transfers even when the boxed answer doesn't.

Use the ``compare_answer`` rules from the Kaggle metric:
  - pure binary string -> exact lowercase compare
  - else if both float-parseable -> math.isclose(rel_tol=1e-2, abs_tol=1e-5)
  - else -> case-insensitive string compare

Usage (current baseline behaviour):
    python scripts/build_sft_traces.py

Usage (matches THK's full-coverage corpus shape while keeping a local holdout):
    python scripts/build_sft_traces.py --keep-wrong-traces --use-reasoner-boxed --holdout 50

Usage (THK corpus + upsample the data-starved category by shuffling
example order; each shuffle produces a distinct reasoner CoT):
    python scripts/build_sft_traces.py --keep-wrong-traces --use-reasoner-boxed \\
        --holdout 50 --upsample equation_numeric_guess:6
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

# Make `from reasoners import ...` work when run as
# `python scripts/build_sft_traces.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reasoners import COMPACT_GENERATORS, GENERATORS  # noqa: E402
from reasoners.store_types import build_problem  # noqa: E402

# Default category set for --drop-wrong-hard-categories. Picked from the
# per-category boxed-correctness audit in docs/next_round_experiment_matrix.md
# rebuild diagnostics: these categories have reasoner-label correctness
# substantially below the rest of the corpus, so keeping their wrong-boxed
# traces actively teaches the model the wrong pattern. bit_manipulation is
# excluded because its label correctness is high (~85%); its bottleneck is
# trace length, not label quality (handled by --compact-bit-manipulation).
DEFAULT_DROP_WRONG_HARD_CATEGORIES = (
    "cryptarithm_deduce",
    "cryptarithm_guess",
    "cryptarithm",
    "equation_numeric_guess",
)


_BOXED_RE = re.compile(r"\\boxed\{([^}]*)(?:\}|$)")

# THK's `PROMPT_SUFFIX` from nemotron-tonghuikang-source/corpus.py:38-41.
# Appended to every reasoner-record user prompt at training time so the model
# sees the same prompt shape that THK's submission notebook produces at
# inference. NOT applied to augmenter records (THK's corpus.py:245 passes
# suffix="" for augmenter prompts).
THK_PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


def shuffle_example_order(prompt: str, seed: int) -> str | None:
    """Return a copy of `prompt` with the example lines reshuffled.

    The Kaggle prompt layout for equation_numeric / cryptarithm tasks is::

        <one-line intro that ends with "examples:">
        <example 1>
        ...
        <example N>
        Now, ...

    We shuffle lines [1:now_idx] using ``seed`` and rejoin. Returns ``None``
    when the structure isn't recognised or there are <2 example lines.
    """
    lines = prompt.splitlines()
    now_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("Now,"):
            now_idx = i
            break
    if now_idx < 2:  # need at least intro + 1 example + Now-line
        return None
    intro = lines[:1]
    examples = lines[1:now_idx]
    rest = lines[now_idx:]
    if len(examples) < 2:
        return None
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    if shuffled == examples:  # nothing actually changed
        return None
    return "\n".join(intro + shuffled + rest)


def extract_boxed(text: str) -> str:
    matches = _BOXED_RE.findall(text)
    if not matches:
        return ""
    non_empty = [m.strip() for m in matches if m.strip()]
    if non_empty:
        return non_empty[-1]
    return matches[-1].strip()


def restructure_for_thinking(trace: str, answer: str) -> str:
    """Reshape a trace as a complete `<think>...</think>\\boxed{}` assistant turn.

    Nemotron 3 Nano's chat template (audited against the live tokenizer
    config) does *not* auto-prepend ``<think>`` when an assistant message
    already contains ``</think>`` -- the content emits verbatim. So the
    SFT data itself has to include the opening tag, otherwise the
    full-text render and the inference prompt prefix don't share a
    common prefix and ``train_lora.py``'s prompt-length-boundary mask
    points at the wrong token index.

    Resulting assistant content:

        <think>
        <reasoning body>
        </think>

        \\boxed{<answer>}

    With this shape, the inference prompt (which ends in
    ``<|im_start|>assistant\\n<think>\\n``) is a true prefix of the
    training render, so masking and supervision align cleanly.

    ``answer`` is the value used in the final ``\\boxed{}``; callers
    decide whether that's the ground truth (baseline) or the reasoner's
    own ``\\boxed{}`` value (``--use-reasoner-boxed``, matches THK).
    """
    lines = trace.splitlines()
    while lines and ("\\boxed{" in lines[-1] or not lines[-1].strip()):
        lines.pop()
    body = "\n".join(lines).rstrip()
    return f"<think>\n{body}\n</think>\n\n\\boxed{{{answer}}}"


def compare_answer(stored: str, predicted: str) -> bool:
    stored = stored.strip()
    predicted = predicted.strip()
    if re.fullmatch(r"[01]+", stored):
        return predicted.lower() == stored.lower()
    try:
        return math.isclose(
            float(stored), float(predicted), rel_tol=1e-2, abs_tol=1e-5
        )
    except Exception:
        return predicted.lower() == stored.lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classified", default="data/problems_classified.jsonl")
    ap.add_argument("--out-train", default="data/sft_traces.jsonl")
    ap.add_argument("--out-holdout-dir", default="data")
    ap.add_argument("--holdout", type=int, default=200,
                    help="Per-category holdout size")
    ap.add_argument("--max-train-per-category", type=int, default=None,
                    help="Cap training samples per category (after holdout)")
    ap.add_argument("--only-category", default=None,
                    help="If set, process only this category")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no-thinking-wrap",
        action="store_true",
        help="Emit the raw trace verbatim (legacy shape). Default wraps "
             "the trace as the post-`<think>` portion of the assistant turn, "
             "matching Nemotron's enable_thinking=True chat template.",
    )
    ap.add_argument(
        "--keep-wrong-traces", action="store_true",
        help="Keep reasoner traces whose boxed answer does NOT match ground "
             "truth. Matches THK's 04-10-04-33 behaviour. Default: drop them.",
    )
    ap.add_argument(
        "--use-reasoner-boxed", action="store_true",
        help="Use the reasoner's own boxed value in the trained \\boxed{...} "
             "(rather than the ground-truth answer). Matches THK's "
             "04-10-04-33 behaviour. Default: rewrite to ground truth.",
    )
    ap.add_argument(
        "--upsample", action="append", default=[],
        metavar="CAT:K",
        help="Generate K total traces per problem in CAT by shuffling the "
             "example-line order in the user prompt (the reasoner branches "
             "on first-example, so different orderings yield distinct CoTs). "
             "Repeatable. Example: --upsample equation_numeric_guess:6 brings "
             "86 base train rows to ~516. The first trace per problem uses "
             "the original prompt; additional ones use deterministic shuffles "
             "with id suffix '__p{i}'.",
    )
    suffix_group = ap.add_mutually_exclusive_group()
    suffix_group.add_argument(
        "--append-thk-suffix",
        dest="append_thk_suffix",
        action="store_true",
        help="Append THK's `Please put your final answer inside \\boxed{}. "
             "For example: \\boxed{your answer}` suffix to every reasoner "
             "record's user prompt, matching THK's corpus.py. This is now "
             "the default because configs/eval_kaggle.yaml appends the same "
             "suffix at inference time. NOT applied to augmenter records.",
    )
    suffix_group.add_argument(
        "--no-append-thk-suffix",
        dest="append_thk_suffix",
        action="store_false",
        help="Deliberately omit THK's prompt suffix from reasoner records. "
             "Use only for ablations where eval-time suffixing is also disabled.",
    )
    ap.set_defaults(append_thk_suffix=True)
    ap.add_argument(
        "--compact-bit-manipulation",
        action="store_true",
        help="Use the compact bit_manipulation reasoner (~470 tokens/trace) "
             "instead of the verbose THK one (~2900 tokens/trace). Lower per-"
             "trace label correctness (~60%% vs 85%%) but the shorter traces "
             "are within rank-32 LoRA cloning capacity, lifting eval accuracy "
             "from the current ~4%% ceiling. Other categories use their "
             "verbose generators unchanged.",
    )
    ap.add_argument(
        "--drop-wrong-categories",
        action="append",
        default=[],
        metavar="CAT",
        help="Drop reasoner traces whose boxed value does NOT match ground "
             "truth, for the listed category. Useful with --keep-wrong-traces "
             "to remove noisy supervision from specific hard categories while "
             "preserving the THK 'procedure over correctness' policy elsewhere. "
             "Repeatable. Has no effect if --keep-wrong-traces is off (wrong "
             "traces are dropped globally in that case).",
    )
    ap.add_argument(
        "--drop-wrong-hard-categories",
        action="store_true",
        help="Convenience shortcut equivalent to passing --drop-wrong-categories "
             "for cryptarithm_deduce, cryptarithm_guess, equation_numeric_guess "
             "(and the legacy combined cryptarithm key). These are the "
             "categories whose reasoner-label correctness is <20%%; keeping "
             "their wrong-boxed traces under THK's policy teaches the model "
             "to clone the wrong answer pattern. bit_manipulation is NOT "
             "included — its label correctness is high; its issue is trace "
             "length (use --compact-bit-manipulation instead).",
    )
    args = ap.parse_args()

    drop_wrong_categories: set[str] = set(args.drop_wrong_categories)
    if args.drop_wrong_hard_categories:
        drop_wrong_categories.update(DEFAULT_DROP_WRONG_HARD_CATEGORIES)
    if drop_wrong_categories and not args.keep_wrong_traces:
        print(
            "[info] --drop-wrong-categories is a no-op without "
            "--keep-wrong-traces (wrong traces are dropped globally)."
        )

    active_generators = (
        COMPACT_GENERATORS if args.compact_bit_manipulation else GENERATORS
    )
    if args.compact_bit_manipulation:
        print("[info] compact bit_manipulation reasoner enabled "
              "(~470 tokens/trace vs ~2900 verbose)")
    if drop_wrong_categories:
        print(f"[info] dropping wrong-boxed traces for categories: "
              f"{sorted(drop_wrong_categories)}")

    upsample_map: dict[str, int] = {}
    for spec in args.upsample:
        if ":" not in spec:
            raise SystemExit(f"--upsample expects CAT:K, got {spec!r}")
        cat, k_str = spec.rsplit(":", 1)
        k = int(k_str)
        if k < 1:
            raise SystemExit(f"--upsample K must be >= 1, got {k}")
        upsample_map[cat] = k

    rows: list[dict] = []
    with open(args.classified, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    by_cat: dict[str, list[dict]] = {}
    for row in rows:
        by_cat.setdefault(row["category"], []).append(row)

    print(f"Loaded {len(rows)} classified problems across {len(by_cat)} categories:")
    for cat, items in sorted(by_cat.items()):
        gen_marker = (
            "(generator)" if cat in active_generators else "(no generator yet)"
        )
        print(f"  {cat:30} {len(items):>6}  {gen_marker}")
    print()

    cats_to_process = (
        [args.only_category] if args.only_category else sorted(by_cat.keys())
    )

    rng = random.Random(args.seed)
    train_records: list[dict] = []
    rule_found_total = 0
    rule_unknown_total = 0
    wrong_answer_total = 0
    no_generator_total = 0

    for cat in cats_to_process:
        items = by_cat.get(cat, [])
        if not items:
            print(f"[{cat}] no rows in classified file, skipping")
            continue

        # Deterministic per-category shuffle so train/holdout split is stable
        # across runs and disjoint across categories.
        local_rng = random.Random(args.seed + (hash(cat) & 0xFFFFFFFF))
        items_shuffled = list(items)
        local_rng.shuffle(items_shuffled)

        holdout_n = min(args.holdout, len(items_shuffled))
        holdout_items = items_shuffled[:holdout_n]
        train_items = items_shuffled[holdout_n:]
        if args.max_train_per_category is not None:
            train_items = train_items[: args.max_train_per_category]

        # Always write the holdout, even if no generator yet.
        holdout_path = Path(args.out_holdout_dir) / f"{cat}_holdout.jsonl"
        holdout_path.parent.mkdir(parents=True, exist_ok=True)
        with open(holdout_path, "w", encoding="utf-8") as f:
            for r in holdout_items:
                f.write(json.dumps({
                    "id": r["id"],
                    "prompt": r["prompt"],
                    "expected_answer": r["answer"],
                }, ensure_ascii=False) + "\n")
        print(f"[{cat}] holdout {len(holdout_items):>5} -> {holdout_path}")

        gen = active_generators.get(cat)
        if gen is None:
            no_generator_total += len(train_items)
            print(f"[{cat}] no generator -- skipping trace generation "
                  f"({len(train_items)} would-be train rows)")
            continue

        cat_found = 0
        cat_unknown = 0
        cat_wrong = 0
        cat_kept_wrong = 0
        cat_upsampled = 0
        cat_upsample_skips = 0
        drop_wrong_here = cat in drop_wrong_categories
        upsample_k = upsample_map.get(cat, 1)
        for r in train_items:
            problem = build_problem(r)
            try:
                trace = gen(problem)
            except Exception as e:
                print(f"[{cat}] {r['id']}: generator error: {type(e).__name__}: {e}")
                trace = None
            if trace is None:
                cat_unknown += 1
                continue
            reasoner_boxed = extract_boxed(trace)
            is_correct = compare_answer(problem.answer, reasoner_boxed)
            if not is_correct:
                if not args.keep_wrong_traces or drop_wrong_here:
                    cat_wrong += 1
                    continue
                cat_kept_wrong += 1

            # Choose what value to put in the trained \boxed{...}.
            if args.use_reasoner_boxed and reasoner_boxed:
                boxed_target = reasoner_boxed
            else:
                boxed_target = problem.answer

            content = (
                trace if args.no_thinking_wrap
                else restructure_for_thinking(trace, boxed_target)
            )
            user_content = r["prompt"]
            if args.append_thk_suffix:
                user_content = user_content + THK_PROMPT_SUFFIX
            train_records.append({
                "id": r["id"],
                "category": cat,
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": content},
                ],
            })
            if is_correct:
                cat_found += 1

            if upsample_k > 1:
                base_id = r["id"]
                seen_prompts: set[str] = {r["prompt"]}
                made = 0
                attempt = 0
                # Cap attempts so a problem with too-few examples (so the
                # permutation space is exhausted) can't hang the loop.
                while made < upsample_k - 1 and attempt < (upsample_k - 1) * 4:
                    attempt += 1
                    shuffled_prompt = shuffle_example_order(
                        r["prompt"],
                        seed=(args.seed ^ (hash(base_id) & 0xFFFFFFFF)) + attempt,
                    )
                    if shuffled_prompt is None or shuffled_prompt in seen_prompts:
                        continue
                    seen_prompts.add(shuffled_prompt)
                    new_row = {**r, "prompt": shuffled_prompt}
                    problem2 = build_problem(new_row)
                    try:
                        trace2 = gen(problem2)
                    except Exception as e:
                        print(f"[{cat}] {base_id}__p{made+1}: generator error: "
                              f"{type(e).__name__}: {e}")
                        cat_upsample_skips += 1
                        continue
                    if trace2 is None:
                        cat_upsample_skips += 1
                        continue
                    reasoner_boxed2 = extract_boxed(trace2)
                    is_correct2 = compare_answer(problem2.answer, reasoner_boxed2)
                    if not is_correct2 and (
                        not args.keep_wrong_traces or drop_wrong_here
                    ):
                        cat_upsample_skips += 1
                        continue
                    if args.use_reasoner_boxed and reasoner_boxed2:
                        boxed_target2 = reasoner_boxed2
                    else:
                        boxed_target2 = problem2.answer
                    content2 = (
                        trace2 if args.no_thinking_wrap
                        else restructure_for_thinking(trace2, boxed_target2)
                    )
                    made += 1
                    up_user_content = shuffled_prompt
                    if args.append_thk_suffix:
                        up_user_content = up_user_content + THK_PROMPT_SUFFIX
                    train_records.append({
                        "id": f"{base_id}__p{made}",
                        "category": cat,
                        "messages": [
                            {"role": "user", "content": up_user_content},
                            {"role": "assistant", "content": content2},
                        ],
                    })
                    cat_upsampled += 1

        rule_found_total += cat_found
        rule_unknown_total += cat_unknown
        wrong_answer_total += cat_wrong
        kept_wrong_note = (
            f", kept_wrong {cat_kept_wrong}" if cat_kept_wrong else ""
        )
        upsample_note = (
            f", upsampled +{cat_upsampled} (skipped {cat_upsample_skips})"
            if upsample_k > 1 else ""
        )
        denom = max(1, len(train_items))
        print(
            f"[{cat}] rule_found {cat_found}/{len(train_items)} "
            f"({100.0 * cat_found / denom:.1f}%); "
            f"unknown {cat_unknown}, wrong {cat_wrong}{kept_wrong_note}"
            f"{upsample_note}"
        )

    # Interleave categories so SFT batches see a mix.
    rng.shuffle(train_records)

    out_train = Path(args.out_train)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    with open(out_train, "w", encoding="utf-8") as f:
        for rec in train_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print()
    print(f"Total training records: {len(train_records)} -> {out_train}")
    print(f"  rule_found (correct boxed): {rule_found_total}")
    if args.keep_wrong_traces:
        print(f"  kept-wrong (procedure only): "
              f"{len(train_records) - rule_found_total}")
    print(f"Skipped (no generator): {no_generator_total}")
    print(f"Skipped (rule_unknown / generator returned None): {rule_unknown_total}")
    print(f"Skipped (wrong answer, dropped): {wrong_answer_total}")


if __name__ == "__main__":
    main()
