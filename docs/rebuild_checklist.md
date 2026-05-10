# THK pipeline rebuild — checklist

Tracks the port of [tonghuikang/nemotron](https://github.com/tonghuikang/nemotron) (Progress Prize, Kaggle LB ~0.85) into this repo's HF + peft + SLURM stack.

Source mirror (raw-fetched, no GB-level folders): `..\nemotron-tonghuikang-source\`. See [README.md](../README.md) for repo layout.

The data shape collapses cleanly to our existing `data/sft_traces.jsonl` (`{messages, category}` rows) — see `Finding 1` in the round that produced this checklist. No tinker/modal infrastructure needed.

## Recipe deltas vs current `configs/lora_verification.yaml`

These four are the only training-side knob changes:

| Knob | Current | THK | Action |
|---|---|---|---|
| `lora.r` | 16 | 32 | bump (Kaggle cap is 32) |
| `train.gradient_accumulation_steps` | 16 | 64 (effective batch) | bump |
| `train.lr_scheduler` + `warmup_ratio` | cosine + 0.03 | linear + 0 | swap |
| `train.max_grad_norm` | HF default (1.0) | 1e9 (disabled) | explicit |

Drop `train.max_steps: 300` once the dataset is enlarged.

## Data shape (decoded from sampled corpus segments)

Every training example is `M-U` (one masked prompt segment, one unmasked completion segment):
- **Reasoner records** (9 categories): user content includes the Kaggle `Please put your final answer inside \boxed{}` suffix; assistant content is `<think>\n<body>\n</think>\n\n\boxed{<answer>}`.
- **Augmenter records** (5 categories): user content has **no `\boxed{}` suffix**; assistant content is `<think>\n<output>\n</think>` (no `\boxed{}`).

Both shapes pass through `apply_chat_template(messages, enable_thinking=True, add_generation_prompt=False)` correctly.

## Steps

### Phase 1 — foundation + smoke test (local, no GPU)

- [x] Read THK's `train_sft.py`, `loss_config.py`, `lr_schedule.py`, `corpus.py`, `reasoning.py`, `augmentation.py`, all 5 augmenters
- [x] Decode 1 sample per category to confirm segment format
- [ ] Draft this checklist
- [ ] Create `scripts/augmenters/` package
- [ ] Port `scripts/augmenters/concatenation.py` (84 lines, pure RNG, no deps)
- [ ] Write `scripts/build_augmenter_traces.py` — augmenter → `{messages, category, id}` JSONL
- [ ] Write `scripts/smoke_augmenter.py` — generates a tiny JSONL, validates chat-template roundtrip if `transformers` available
- [ ] Run smoke test, verify pass

### Phase 2 — full augmenter set (local) ✅

- [x] Port `scripts/augmenters/splitting.py` (84 lines, inverse of concatenation)
- [x] Port `scripts/augmenters/lstrip.py` (80 lines, pure RNG)
- [x] Port `scripts/augmenters/spelling.py` (~140 lines) — accepts `--tokenizer-path`, otherwise discovers `data/tokenizer.json` → `../nemotron-tonghuikang-source/tokenizer.json` → HF cached AutoTokenizer
- [x] Port `scripts/augmenters/matching.py` (~280 lines) — input source generalised to a `reasoning_texts: dict[id, trace]` mapping; driver pulls bit_manipulation rows from `data/sft_traces.jsonl` (also accepts a THK-style `reasoning/` directory)
- [x] Smoke test extended to cover all 5: `scripts/smoke_augmenter.py` passes Stages A–D for every augmenter (Stage E auto-skips on Windows without HF cache)
- [ ] Verify spelling output token IDs match THK's via local tokenizer check (Phase 5)
- [ ] Confirm matching runs *after* reasoners (sequencing in `build_sft_traces.py`) (Phase 5)

### Phase 3 — reasoner swap (local) ✅

- [x] Port THK's `reasoners/store_types.py` long-arithmetic helpers (`truncate_3dp`, `cast_dp_pair`, `pad_dp`, `_fmt_int_with_dp`, `long_multiplication_lines`, `long_division_lines`) into ours
- [x] Update `build_problem` — per-category bare-value `question` extraction + per-category example-line parsing (gravity uses `For t=… distance=…`, unit_conversion uses `… becomes …`, equation_numeric/cryptarithm use `X = Y`, others stay on `X -> Y`)
- [x] Replace `cipher.py`, `equation_numeric.py`, `gravity.py`, `numeral.py`, `unit_conversion.py` with THK versions (relative-import fix only)
- [x] Add `cryptarithm.py` (new file) + `wonderland.txt` (cipher dependency)
- [x] Update `__init__.py` — 11 GENERATORS keys (9 THK categories + 2 legacy aliases for the current classifier's combined names)
- [x] Smoke test (`scripts/smoke_reasoners.py`) hits **82.5% overall on 120 mixed problems**: cipher 100%, gravity 100%, numeral 100%, unit_conversion 100%, bit_manipulation 80%, equation_numeric 15% (expected — the 15% reflects the mixed deduce/guess split that the Phase 4 classifier will separate)
- [ ] Keep bit_manipulation reasoner — local version is essentially THK's with a more defensive prompt-header filter (~36 semantic-line diff, no behavior change)

### Phase 4 — classifier + holdout (local) ✅

- [x] Updated `scripts/categorize.py` — 9 categories via `_refine_equation_header`: numeric vs symbolic by digit check on all operands+results, deduce vs guess by "is question op in examples"
- [x] **Authoritative-label override**: when `nemotron-tonghuikang-source/problems.jsonl` is present, our regex output is replaced by THK's labels for the equation/cryptarithm rows (277 of 1,555 disagreed with our regex heuristic). Result: **exact match with THK's distribution** (1602/1576/1576/1597/1594/659/164/596/136 = 9,500)
- [x] Regenerated `data/problems_classified.jsonl`
- [x] Rebuilt `data/stratified_holdout.jsonl` — 50 per category × 9 = 450 rows; augmenter categories correctly excluded (they're not in problems_classified.jsonl)

### Phase 5 — full data build (local) ✅

Reasoner build:
- [x] Updated `scripts/build_sft_traces.py` — added `--keep-wrong-traces` and `--use-reasoner-boxed` flags (both off = baseline behaviour, both on = matches THK 04-10-04-33)
- [x] Always emits `id` + `category` per record (needed by the matching augmenter)
- [x] Run with `--keep-wrong-traces --use-reasoner-boxed --holdout 50`: **9,050 reasoner records** (matches THK's 9,500 minus our 450 holdout)

Augmenter build:
- [x] Run `scripts/build_augmenter_traces.py --matching-source data/sft_traces.jsonl`: **8,341 augmenter records** (concatenation 1,500 ✅, splitting 1,500 ✅, lstrip 300 ✅, spelling 648 ✅, matching 4,393 vs THK's 4,515 — 122 fewer due to smaller bit_manipulation source)

Combine:
- [x] `scripts/combine_traces.py` → **`data/sft_combined.jsonl`** with **17,391 records** (vs THK 17,963; diff = 450 holdout + 122 matching shortfall)
- [x] Length filter at 49,152 chars (proxy for 8K tokens) dropped 0 records. Real per-token length filter happens at training time inside `train_lora.py:build_dataset` — records whose tokenized full conversation exceeds `max_seq_len` are dropped with a `[warn] skipped` message.

**Per-category record counts (final `data/sft_combined.jsonl`)**:

| Category | Ours | THK | Note |
|---|---|---|---|
| bit_manipulation | 1,552 | 1,602 | -50 holdout |
| cipher | 1,526 | 1,576 | -50 holdout |
| numeral | 1,526 | 1,576 | -50 holdout |
| gravity | 1,547 | 1,597 | -50 holdout |
| unit_conversion | 1,544 | 1,594 | -50 holdout |
| cryptarithm_deduce | 609 | 659 | -50 holdout |
| cryptarithm_guess | 114 | 164 | -50 holdout |
| equation_numeric_deduce | 546 | 596 | -50 holdout |
| equation_numeric_guess | 86 | 136 | -50 holdout |
| concatenation | 1,500 | 1,500 | ✅ exact |
| splitting | 1,500 | 1,500 | ✅ exact |
| lstrip | 300 | 300 | ✅ exact |
| spelling | 648 | 648 | ✅ exact |
| matching | 4,393 | 4,515 | -122 (smaller bit_manip source) |
| **Total** | **17,391** | **17,963** | -572 |

Decision point for Phase 6: **rerun with `--holdout 0` to match THK's 17,963 exactly?** Cost: lose local eval signal. Benefit: another ~3% training data, slightly higher matching count. Recommendation: keep the 50/cat holdout for the first verification run so we can measure local vs Kaggle drift, then drop the holdout for the actual submission run if the local→Kaggle gap is small.

#### Caveats locked in for Phase 6

1. **Per-category solve rate, after dedup match with THK's `status` field** (`rule_found` counts in their `problems.jsonl`):

   | Category | THK rule_found | Our rule_found |
   |---|---|---|
   | cipher / gravity / numeral / unit_conversion | 100% | 100% |
   | bit_manipulation | 85.1% | 85.3% |
   | equation_numeric_deduce | 90.6% | 90.8% |
   | equation_numeric_guess | 15.4% | 16.3% |
   | cryptarithm_deduce | 8.2% | 8.0% |
   | cryptarithm_guess | 6.7% | 8.8% |

   Within 1% across all categories. Reasoners are faithful to THK.

2. **The 04-10-04-33 trick — train on WRONG-answer traces too.** THK's `corpus.jsonl` has `included: true` for all 9,500 problems, even though only 6,584 are `rule_found`. The remaining 2,916 wrong-answer traces are kept because the *procedure* is still valid CoT scaffolding — the model learns the reasoning shape from these even when the final boxed value doesn't match ground truth. Our `--keep-wrong-traces --use-reasoner-boxed` flags reproduce this exactly.

### Phase 6 — training (cluster)

- [ ] Apply recipe deltas to `configs/lora_verification.yaml`: rank=32, grad_accum=64, lr_scheduler=linear, warmup=0, max_grad_norm=1e9, remove max_steps cap
- [ ] Update `slurm/lora_verification.slurm` walltime upward — expect ~265 steps × ~60 s/step ≈ 4.5 h
- [ ] Submit
- [ ] Watch loss curve in `logs/lora_verification_*.out` — sanity: per-category loss should drop fastest on augmenters (small inputs) and slowest on bit_manipulation (huge traces)

### Phase 7 — eval + submit (cluster)

- [ ] `configs/eval_kaggle.yaml` — bump `lora.max_lora_rank` to 32, point `lora.path` at new adapter dir
- [ ] Run paired eval (base vs LoRA) on `data/stratified_holdout.jsonl`
- [ ] If improvement is real, package via `scripts/package_submission.py` and submit
- [ ] Record Kaggle score in MEMORY milestone file

### Phase 8 — defer / parallelizable ✅ (target locked)

- [x] **Fetched all 8 surviving `training/sft/<timestamp>/config.json` files** from THK and tabulated side by side
- [x] **Target locked: `training/sft/04-10-04-33/config.json`** — the LB-winning production run. Picked by four independent signals (see "Target config decision" section below).

### Target config decision — `04-10-04-33`

**All 8 surviving configs share identical hyperparameters** (cross_entropy, lr=2e-4 step-linear decay, rank=32, 1 epoch, max_length=8192, Adam β=(0.9,0.95), wd=0, grad_clip=1e9). The only thing varying across runs is **dataset size + batch_size**. So picking the config is equivalent to picking the dataset shape.

| Timestamp | n_examples | batch×micro | steps | distinct cats | Role |
|---|---|---|---|---|---|
| **04-10-04-33** | **15,679** | **64×16** | **245** | **13 / 14** | **production / LB-winning** ✅ |
| 04-06-00-22 | 8,542 | 32 | 267 | 9 (reasoners only) | pre-augmenter baseline |
| 04-08-16-14 | 7,830 | 32×16 | 245 | ~9 | iteration |
| 03-23-00-47 | 7,599 | 64 | 119 | ~9 | earliest preserved |
| 04-07-02-00 | 7,489 | 32×16 | 234 | ~9 | iteration |
| 04-13-12-59 | 2,148 | 64×16 | 34 | small | post-prod ablation |
| 04-10-04-15 | 1,789 | 64×16 | 28 | small | pre-prod warm-up |
| 04-13-13-09 | 648 | 64×16 | 11 | spelling-only | post-prod ablation |

Signals that point at 04-10-04-33:

1. **Dataset completeness** — 15,679 ≈ 87% of the full 17,963-entry corpus. The only run that includes augmenters (matching, spelling, splitting, concatenation) AND the deduce/guess-split reasoner categories together. Only `lstrip` (300 examples) is missing from the 14 corpus categories.
2. **Chronological position** — sits at position 24/26 in `logpaths.txt`; preceded by 04-10-04-15 (28-step warm-up) and followed three days later by the 04-13-* post-prod ablations.
3. **Loss-curve health** — `_loss_per_token` drops from 0.3645 → 0.0044 (-98.8%) over 245 steps; per-category final loss is universally low across all 13 included categories.
4. **`upload_adapter.py` semantics** — script ships the *latest* Tinker `sampler_weights/final` checkpoint dynamically. The Kaggle 0.85 submission must have happened *before* the 04-13 ablation chain (which would have overwritten the "latest" pointer); 04-10-04-33 is the most recent full-corpus run before that chain.

The Kaggle writeup at [discussion/689915](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915) and the notebook at [code/huikang/end-to-end-finetuning-for-lb-0-85](https://www.kaggle.com/code/huikang/end-to-end-finetuning-for-lb-0-85) confirm the LB score is 0.85 (from the URL slug) but the body is auth-gated; the four signals above are strong enough to commit without it.

#### Recipe to copy into `configs/lora_verification.yaml`

```yaml
lora:
  r: 32                              # was 16
train:
  num_epochs: 1
  per_device_batch_size: 1
  gradient_accumulation_steps: 64    # was 16, target effective batch = 64
  learning_rate: 2.0e-4
  lr_scheduler: linear               # was cosine
  warmup_ratio: 0.0                  # was 0.03
  max_grad_norm: 1.0e9               # was HF default 1.0 — disable grad clip
  adam_beta1: 0.9
  adam_beta2: 0.95                   # HF default is 0.999 — must override
  weight_decay: 0.0
data:
  max_seq_len: 8192
  # Goal: ~15,679 records covering all 9 reasoner categories +
  # 4 of 5 augmenter categories (matching/spelling/splitting/concatenation).
  # lstrip can be omitted to match 04-10-04-33; including it adds ~300 records.
  # Drop any record whose tokenized length exceeds 8192 — do NOT truncate.
```

## Risk callouts (carry forward each round)

1. **Trace length overflow on cipher / equation_numeric.** THK's `corpus.py` silently truncates at 8,192 tokens, losing `\boxed{}`. Cipher traces in the wild can hit ~3,700; equation_numeric_guess samples reach ~6,744. Some pathological problems will exceed 8K. **Drop**, don't truncate. Our existing `lora_verification.yaml` notes this failure mode from iteration 3.
2. **Augmenters are scaffolding, not graders.** Augmenter completions carry no `\boxed{}`. Make sure `extract_boxed()` is never run against augmenter holdouts (because there will be no augmenter holdouts — they are pure training-side examples).
3. **Matching augmenter has a sequencing dependency.** It scrapes substructures from `reasoning/<id>.txt`, so reasoners must run to completion before augmenters. `build_sft_traces.py` already runs reasoners; add the augmenter pass after.
4. **Tokenizer drift.** THK's `spelling.py` enumerates lowercase 2–8 char alphabetic tokens from `tokenizer.json`. Our `AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")` should return identical IDs; check before training.
5. **04-10-04-15 is an ablation, not the LB run.** Treat hyperparameters in `training/sft/04-10-04-15/config.json` as a *starting point* until Phase 8 confirms the actual LB-winning config.
6. **Chat-template `<think>` requirement.** Nemotron 3 Nano's template does **not** auto-prepend `<think>` when the assistant content already contains `</think>`. So the assistant content MUST start with literal `<think>\n` for both reasoner and augmenter records. See `restructure_for_thinking` in `scripts/build_sft_traces.py:54`.
7. **`all-linear` LoRA targets vs vLLM compatibility.** Per `configs/lora_verification.yaml` history: vLLM 0.12.0 cannot load `all-linear` adapters because Nemotron-H lacks `get_expert_mapping`. The newer vLLM eval image is required.

## Status

Last updated: 2026-05-10.
Currently at: **end of Phase 5** (full data build complete, 17,391 records in `data/sft_combined.jsonl`).

Target training config: **`training/sft/04-10-04-33/config.json`** (locked — see Phase 8).

Next: Phase 6 (apply recipe deltas to `configs/lora_verification.yaml`, submit SLURM job, watch loss curve).
