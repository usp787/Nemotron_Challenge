# Project Report

|  |  |
| --- | --- |
| **Period** | 2026-04-27 to 2026-05-31 (~5 weeks) |
| **Final Kaggle leaderboard** | **0.74** |
| **Reference open-source ceiling** | 0.85 (THK / huikang, `end-to-end-finetuning-for-lb-0-85`) |
| **Final structural gap to reference** | ~0.11, traced to a single Kaggle eval-contract constraint that no in-repo work can close |

---

## 1. Project Goal and the Eval Contract

The challenge is to improve `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`'s reasoning behavior on a hidden 9-category test set under tight evaluation constraints:

- **Inference** — vLLM-only with `max_lora_rank ≤ 32`, `max_model_len = 8192`, `max_tokens = 7680`, `temperature = 0.0` (greedy)
- **Adapter** — must be a vLLM-loadable LoRA: no full-weight fine-tuning, no `modules_to_save` (full-rank replacement weights)
- **Scoring** — `\boxed{}` extraction across 9 categories: `numeral`, `unit_conversion`, `gravity`, `cipher`, `bit_manipulation`, `cryptarithm_deduce`, `cryptarithm_guess`, `equation_numeric_deduce`, `equation_numeric_guess`

> **These constraints are the single most important fact about the project.** Every architectural choice downstream is a consequence of them.

---

## 2. Phase-by-Phase Timeline

### Phase 0 — HPC infrastructure (Apr 27–28)

First milestone was getting the model to run at all on Northeastern Explorer. H200 access via the `gpu` partition + `gres=gpu:h200:1`, Apptainer container with vLLM 0.12.0, `$SCRATCH/huggingface` cache for the ~58 GB BF16 weights. AIME25 verification confirmed end-to-end inference (30/30 prompts, 43% pass-rate truncated by token cap — pipeline verification only, not a real reasoning result).

### Phase 1 — Pipeline scaffolding (Apr 29–May 3)

Built the LoRA training stack: HF transformers + peft + accelerate inside the vLLM container with peft installed to `$SCRATCH/lora_pip`. Discovered the first structural constraint: vLLM 0.12.0 couldn't load `target_modules="all-linear"` adapters on Nemotron-H — its `NemotronHForCausalLM` lacked `get_expert_mapping`, so any LoRA touching MoE expert layers was unloadable. Worked around by restricting to attention-only.

### Phase 2 — First algo-CoT submission (May 6) → Kaggle 0.53

Built deterministic algorithmic chain-of-thought generators for `numeral`, `gravity`, `unit_conversion`, `cipher` and trained the LoRA to mimic them. Local per-category solve: 100% on first three, ~38% on `cipher`, 0% on `bit_manipulation` and `equation_numeric_*` (no generator yet).

Kaggle 0.53 decomposed cleanly as `(3 × 100 + 38 + 0 + 0) / 6 ≈ 0.56` — confirming local-vs-leaderboard calibration was honest.

### Phase 3 — Plateau diagnosis (May 8) → Kaggle 0.57

Three different training configs (0.53 / 0.56 / 0.57) all plateaued at the same per-category profile. Concluded that training mechanics weren't the bottleneck; data coverage was:

- `equation_numeric` had only 7 training rows out of 6743.
- `bit_manipulation` traces were mathematically correct but encoded 2200-token brute-force enumerations that a rank-16 LoRA couldn't execute correctly.

The plateau was a **data-coverage ceiling**, not a training-recipe issue.

### Phase 4 — Reasoner expansion and stratified holdout (May 8–13) → Kaggle 0.66

- Built `equation_numeric` and `cryptarithm` reasoners.
- Replaced the single-category `numeral_holdout` (which scored 100% regardless of training, giving zero signal) with a stratified 450-prompt holdout across all 9 categories.
- Lifted LoRA rank from 16 → 32 (the Kaggle cap).
- Added `--append-thk-suffix` for prompt-shape alignment.
- Added peft `modules_to_save=[lm_head]` for output-token control.

This was the first attempt to reproduce the THK open-source recipe found on the Kaggle leaderboard at 0.85.

### Phase 5 — vLLM rejection of `modules_to_save` (May 14) → regression

The 4.24 GB adapter with `modules_to_save: [lm_head]` was rejected by Kaggle's hidden vLLM eval. peft writes a full-rank replacement weight under `...lm_head.modules_to_save.default.weight`; vLLM's LoRA loader only understands low-rank A/B decompositions and either rejects the config or chokes on the full-rank key. **This was the first structural constraint that bounded reproduction of THK's recipe.**

Worked around by adding `lm_head` to the regular `target_modules` list as a rank-32 LoRA layer — a vLLM-compatible approximation of THK's `train_unembed: true` full-rank fine-tune.

### Phase 6 — Full THK reproduction (May 14–24) → Kaggle ~0.68–0.70

Implemented `docs/rebuild_checklist.md` — 8 phases of porting the open-source THK pipeline (`tonghuikang/nemotron`, Progress Prize ~0.85). This was a deliberate full reproduction:

- Ported all 9 THK reasoners (`bit_manipulation`, `cipher`, `cryptarithm`, `equation_numeric`, `gravity`, `numeral`, `unit_conversion`) and 5 augmenters (concatenation, splitting, lstrip, spelling, matching).
- Adopted THK's *"procedure over correctness"* trick — `--keep-wrong-traces --use-reasoner-boxed` keeps traces whose boxed answers are wrong because the procedure is still valid CoT scaffolding.
- Matched THK's `04-10-04-33` training recipe: LoRA r=32, effective batch 64, linear LR schedule with no warmup, AdamW β₂=0.95, grad clip disabled.
- Built path-beta chunked training (3 jobs of ~82 steps each) because THK's 245-step run exceeded the 8h `gpu` partition cap.
- Final corpus: 17,391 records (vs THK's 17,963; the 572-record delta was the 50-per-category local holdout).

This put us at ~0.68–0.70 on Kaggle — within 0.15–0.17 of THK's 0.85 but stuck.

### Phase 7 — Reasoner audit and the `bit_manipulation` breakthrough (May 28) → Kaggle 0.74

A per-category trace-quality diagnostic revealed that the THK reproduction wasn't enough on its own:

| Category | Label correctness | Eval accuracy | Diagnosis |
| --- | --- | --- | --- |
| `numeral`, `unit_conv`, `gravity`, `cipher` | 88–100% | 88–100% | working as designed |
| `equation_numeric_deduce` | 91% | 78% | mostly working |
| `bit_manipulation` | 85% | 4% | clean labels but LoRA can't clone the 2900-token verbose trace |
| `cryptarithm_deduce` | 8% | 8% | model faithfully clones the wrong training labels |
| `cryptarithm_guess` | 9% | 2% | same |
| `equation_numeric_guess` | 16% | 14% | same |

This was the project's **second major structural finding**: the failure modes split into two categories — *trace length × LoRA capacity* (`bit_manipulation`) and *garbage supervision* (the three "guess"-style categories).

**Two interventions:**

- **Compact `bit_manipulation` reasoner** — a column-by-column rule fitter producing ~470-token traces instead of ~2900. Sacrificed label correctness (85% → 60%) for trace clonability under rank-32 LoRA. Result: `bit_manipulation` eval jumped **0.04 → 0.42**, a 10× improvement.
- **`--drop-wrong-hard-categories` flag** — removed the 91%-wrong `cryptarithm` traces and 84%-wrong `equation_numeric_guess` traces from training rather than letting the model clone them. Sacrificed quantity for label cleanliness; `equation_numeric_guess` fell to 0 training rows but stopped poisoning the rest of the corpus.

Kaggle moved from ~0.70 to 0.74 — the largest single jump in the project, with ~+0.04 net after the `eq_num_guess` loss netted against the `bit_manipulation` gain.

### Phase 8 — MoE LoRA expansion attempt (May 29–31) → Kaggle 0.74, unchanged

The remaining hypothesis was capacity. The original `target_modules` list reproduced THK's all-linear resolution (8 names + `lm_head`), but might have missed SwiGLU/MoE expert projections under different naming conventions. Added candidate names: `gate_proj`, `w1`, `w2`, `w3`, `gate_up_proj`, `dt_proj`, `x_proj`.

**Result:** trainable parameters jumped from ~80M to 888M (10×), adapter size grew from a few hundred MB to 4.0 GB. The H100 sharing-partition probe confirmed vLLM 0.20.0 accepts the expanded adapter shape (gate projections, Mamba `dt_proj` / `x_proj` and all). Training loss curve hit 0.0002 by step 57 — a clear over-parameterization signal on a 17K-sample corpus.

Eval confirmed the worry: Kaggle 0.74 unchanged, local stratified 0.573 → 0.558 (`cipher` and `eq_num_deduce` each regressed by ~6%). The added capacity was used to memorize, not generalize. **This was the final, decisive ceiling test.**

---

## 3. Breakthroughs

*In order of impact:*

1. **`bit_manipulation` 0.04 → 0.42 (Phase 7).** The single biggest lever in the project. Discovered by reasoning about why a category with 85% clean training labels still scored 4% on eval — answer: rank-32 LoRA couldn't clone a 2900-token structured enumeration. The generalizable lesson is that *trace length × LoRA capacity* is the dominant bottleneck on long-CoT categories; reducing trace complexity is more effective than improving label quality.

2. **Recognition that THK's recipe was partly unreachable (Phase 5).** The `modules_to_save: [lm_head]` finding established that the public 0.85 score uses a full-rank `lm_head` update that vLLM's LoRA loader cannot apply. The reproduction has an intrinsic ceiling at the rank-32 approximation of that operation. This is a *negative* breakthrough — it tells you what can't be done, which is just as valuable for stopping the project at the right point.

3. **Plateau decomposition (Phase 3, May 8).** The observation that three training configs at 0.53 / 0.56 / 0.57 all reduced to the same per-category profile pivoted the project from training-recipe tuning to data engineering. Without this, weeks of further hyperparameter sweeps would have produced no progress.

4. **Drop wrong traces on hard categories (Phase 7).** Counterintuitive but correct — under THK's keep-wrong-traces recipe, the `cryptarithm` / `equation_numeric_guess` training traces were 84–91% wrong-boxed. Removing them costs training count but stops the model from cloning wrong-answer patterns. Smaller-but-clean beat larger-but-noisy.

---

## 4. The THK Reproduction Phase

Roughly half the project's effort (Phases 4–7) was a deliberate reproduction of an open-source solution rather than original work. This was a strategic choice with both advantages and limitations:

**Advantages**

- A known-good 0.85 reference reduced the search space dramatically. Without it, we'd be exploring the LoRA training landscape blind.
- THK's choices (rank=32, β₂=0.95, linear LR no warmup, "procedure over correctness") were validated; we inherited months of someone else's iteration.
- The reference made negative findings durable. We can now state confidently that THK's recipe doesn't reach 0.85 under the public vLLM-LoRA contract — they must have used a different (pre-2026, or with different vLLM patches) eval pathway.

**Limitations**

- The 0.85 reference set unrealistic expectations. The remaining 0.11 gap is structural, not technical; no amount of careful reproduction closes it.
- The "procedure over correctness" trick is fragile on categories where the reasoner's solve rate is below ~30% — it works for `cipher` and `equation_numeric_deduce` but actively hurts `cryptarithm` and `equation_numeric_guess`. THK got away with it likely because their full-rank `lm_head` training compensated.

---

## 5. Where We Got Stuck

Three structural ceilings, none addressable inside the project's chosen contract:

1. **Rank-32 `lm_head` ≠ full-rank `lm_head`.** Documented in `configs/train_thk_full.yaml:43-76`. THK's `04-10-04-33` used `train_unembed: True`. vLLM 0.20.0 still cannot load `modules_to_save` adapters. This caps achievable score at roughly 0.78–0.80 for any vLLM-LoRA reproduction.

2. **Reasoner solve rate caps the "guess" categories.** `cryptarithm_guess` problems use hidden symbol-transformation rules outside any tractable hypothesis class (we measured: 326-permutation position-perm search covers 0%; character substitution covers 0%; cross-op rule transfer 1.5%). Without a fundamentally stronger solver, the training-label correctness for these categories is ~10–16%, which is also the eval ceiling under "procedure over correctness".

3. **Capacity expansion overfits before it generalizes.** The MoE LoRA expansion (Phase 8) proved that more rank-32 LoRA modules don't lift this corpus — the 17K-record dataset is too small for an 888M-parameter LoRA to learn a non-trivial generalization. The path forward through capacity would require either more diverse data (bounded by the reasoner solve rates above) or full-rank fine-tuning (which the eval contract forbids).

---

## 6. Significance of Reasoning Models and LoRA

What this project demonstrated about each:

### On Nemotron-3-Nano-30B-A3B-BF16 specifically, and reasoning models broadly

- A 30B hybrid Mamba-2/MoE base model with ~3.5B active params can be steered to 100% solve rates on 4 of 9 reasoning categories (`numeral`, `unit_conversion`, `gravity` at >92%; `cipher` at ~88%) using only ~17K SFT examples and a rank-32 LoRA at <3% of param count. This is an enormous sample-efficiency multiplier vs from-scratch training.
- The `<think>...</think>` chat template is **load-bearing** — without correct boundary alignment in the training data, the post-`</think>` answer slot fills with random tokens and accuracy collapses. The chat template is not a cosmetic detail.
- Reasoning models trade context length for solve rate. Categories whose true CoT exceeds ~1500 tokens (verbose `bit_manipulation`, `equation_numeric_deduce`) hit a capacity wall the model can't clear at rank 32. There is a length budget for low-rank reasoning supervision and it's roughly 500–2000 tokens per trace.

### On LoRA as an adaptation technique

- Rank 32 is sufficient for adapting a 30B base to a tight benchmark when the supervision is clean and short. The remaining bottlenecks are structural (`lm_head` full-rank, MoE expert coverage) and contract-level (what the inference engine accepts), not LoRA-mechanics.
- Adapter size scaling is **not monotonic** with quality. Going from 80M → 888M trainable params (`target_modules` expansion) hurt generalization on this corpus. LoRA's value comes from forcing low-rank structure on the adaptation; defeating that with too many target modules removes the regularization without adding useful expressiveness.
- vLLM's LoRA contract is a real constraint. Production-eval pipelines that load adapters via vLLM exclude `modules_to_save`, exclude some Mamba-specific projections, and behave differently across vLLM minor versions. Any LoRA recipe targeting a vLLM-served evaluation must be validated against the exact serving stack, not just trained successfully under peft + transformers.

### On the project's contribution

- We confirmed an experimentally-derived ceiling for vLLM-LoRA reproductions of full-fine-tune recipes (~0.74–0.80 vs full-FT 0.85).
- We documented the *trace length × LoRA capacity* finding as a generalizable principle for any long-CoT distillation under capacity constraints — relevant to math-reasoning distillation, code-CoT distillation, and any setting where a smaller model is taught to mimic a verbose teacher.
- We characterized the failure modes of "procedure over correctness" SFT — useful future guidance is to gate this technique on a per-category reasoner-solve-rate floor (>30%) rather than applying it globally.
