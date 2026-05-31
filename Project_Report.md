# Project Report

## Executive Summary

| Item | Current record |
| --- | --- |
| Period | 2026-04-27 to 2026-05-31, about 5 weeks |
| Target model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| Final Kaggle raw leaderboard score | **0.74** |
| Reference open-source score | 0.85, THK / huikang `end-to-end-finetuning-for-lb-0-85` |
| Current interpretation | 0.74 is the best outcome reached so far under the tested vLLM-LoRA route and the ablations/operations described below |
| Evidence policy | Kaggle exposes only the raw score. Large adapters live on the cluster and are not committed because they are several GB. The final local eval log should be attached manually when available. |

This report is a retrospective of what the project has shown so far, not a claim
that every possible route has been exhausted. The strongest conclusion is
operational: within the tested Kaggle-compatible vLLM LoRA workflow, gains came
from data shape, prompt/eval alignment, and hard-category trace policy; additional
adapter capacity did not improve the final score.

The project reached **0.74** on Kaggle after moving from a first algorithmic-CoT
LoRA at 0.53 through a THK-style reproduction and then hard-category repairs.
The remaining gap to the 0.85 reference is best explained by the combination of
tested constraints: the public vLLM LoRA path cannot use THK-style full-rank
`lm_head` / `train_unembed` updates, some "guess" categories have low reasoner
label correctness, and a larger MoE/Mamba target-module adapter overfit rather
than improving generalization in the tested run.

---

## Big-Picture Pipeline

The current project path is:

```text
scripts/categorize.py
  -> scripts/build_sft_traces.py
  -> scripts/build_augmenter_traces.py
  -> scripts/combine_traces.py
  -> slurm/train_thk_chunk.slurm, three path-beta chunks
  -> slurm/eval.slurm
  -> slurm/package.slurm / scripts/package_submission.py
  -> Kaggle raw score
```

The local tracked data snapshot used by the THK-rebuild documentation has:

| Artifact | Tracked repo snapshot |
| --- | ---: |
| `data/sft_traces.jsonl` | 9,050 rows |
| `data/augmenter_traces.jsonl` | 8,341 rows |
| `data/sft_combined.jsonl` | 17,391 rows |
| `data/stratified_holdout.jsonl` | 450 rows |

Later cluster-side runs may use rebuilt data and larger adapter artifacts that
are intentionally not stored in git. The final eval log can be added below as
the durable small artifact for the final score path.

### Evaluation Evidence To Attach

Kaggle does not expose hidden per-category scores, predictions, or logs. The
leaderboard score is therefore treated as trusted but not transparent. The
adapter artifacts are also too large for convenient git storage. The visible
evidence set for the final outcome is expected to be:

- Kaggle raw score: **0.74**.
- Last local eval-script log: **to be pasted manually**.
- Repo configs and scripts documenting the tested pipeline.

---

## Evaluation Contract

The competition setup shaped nearly every design choice:

- **Inference:** vLLM-only serving, `max_lora_rank <= 32`, `max_model_len = 8192`,
  `max_tokens = 7680`, `temperature = 0.0`.
- **Adapter:** Kaggle loads a LoRA adapter. Full fine-tuning weights and
  full-rank `modules_to_save` payloads are not part of the tested working
  submission path.
- **Scoring:** answers are extracted from `\boxed{}` over 9 categories:
  `numeral`, `unit_conversion`, `gravity`, `cipher`, `bit_manipulation`,
  `cryptarithm_deduce`, `cryptarithm_guess`,
  `equation_numeric_deduce`, and `equation_numeric_guess`.

The important framing is "what worked under this contract." Some THK behaviors
appear reproducible in the repo, while others only have a Kaggle-compatible
approximation.

---

## Phase Timeline

### Phase 0: HPC Infrastructure, Apr 27-28

The first milestone was simply making the 30B BF16 model run on Northeastern
Explorer. The working route used H200 GPUs on the `gpu` partition, Apptainer,
the `vllm/vllm-openai:v0.12.0` container, and `$SCRATCH/huggingface` for the
model cache. AIME25 smoke verification established that end-to-end inference
worked. That result was pipeline validation, not a competition-quality result.

### Phase 1: LoRA Pipeline Scaffolding, Apr 29-May 3

The repo gained a HF Transformers + PEFT + Accelerate training path inside the
container. The first adapter topology lesson was that vLLM 0.12.0 could not load
the original `target_modules="all-linear"` Nemotron-H adapter because the model
class lacked `get_expert_mapping` for MoE LoRA. The practical workaround was to
avoid unsupported expert-targeting in the early verification path and later use
the newer vLLM LoRA eval image.

### Phase 2: First Algorithmic-CoT Submission, May 6 -> Kaggle 0.53

The first real submission trained on deterministic algorithmic traces for
`numeral`, `gravity`, `unit_conversion`, and `cipher`. The local profile was
strong on the first three categories, partial on `cipher`, and uncovered on
`bit_manipulation` and equation categories. Kaggle 0.53 was consistent with
that visible profile, which made the local-vs-Kaggle calibration feel credible.

### Phase 3: Plateau Diagnosis, May 8 -> Kaggle 0.57

Several training configs landed in the same 0.53-0.57 region. The repo notes
pointed away from generic hyperparameter tuning and toward data coverage:

- `equation_numeric` had extremely sparse training coverage at that point.
- `bit_manipulation` had long, correct-looking traces that a small rank-16 LoRA
  did not execute reliably.

This was the first major negative result: the bottleneck was not just "train
longer" or "tune learning rate." It was category coverage and trace usability.

### Phase 4: Reasoner Expansion and Stratified Holdout, May 8-13 -> Kaggle 0.66

The project added `equation_numeric` and `cryptarithm` reasoners, moved from the
single-category `numeral_holdout` to a 450-row stratified holdout, raised LoRA
rank to 32, aligned prompt suffixes with THK, and tried `modules_to_save=[lm_head]`
for output-token control.

This phase was the start of the deliberate THK reproduction attempt.

### Phase 5: `modules_to_save` Rejection, May 14

The `modules_to_save=[lm_head]` adapter was rejected by the Kaggle/vLLM path.
PEFT writes a full-rank replacement tensor under the `modules_to_save` key,
whereas the working vLLM LoRA loader expects low-rank A/B matrices. The repo
therefore switched to putting `lm_head` in `target_modules`, which trains a
rank-32 LoRA approximation of THK's `train_unembed: true` behavior.

This is a strong structural finding for the tested submission route. It does
not prove no other private or future serving route could behave differently,
but it does explain why a direct THK full-rank unembedding reproduction is not
available in the current public vLLM-LoRA workflow.

### Phase 6: THK-Style Reproduction, May 14-24 -> Kaggle about 0.68-0.70

The repo ported the THK-style reasoner and augmenter pipeline:

- 9 reasoner categories.
- 5 augmenters: concatenation, splitting, lstrip, spelling, and matching.
- THK's "procedure over correctness" data policy through
  `--keep-wrong-traces --use-reasoner-boxed`.
- The `04-10-04-33` style recipe: LoRA rank 32, effective batch 64, linear LR,
  no warmup, AdamW beta2 0.95, and disabled practical grad clipping.
- Path-beta chunked training because a single 245-step production run did not
  fit inside the `gpu` partition walltime.

The tracked repo snapshot for this reproduction has 17,391 combined records,
which is the holdout-50 build documented in `docs/rebuild_checklist.md`. The
cluster may contain later rebuilt variants; this report treats those as
log-backed operational artifacts rather than git-tracked files.

The THK-style route improved the Kaggle score but did not reach the 0.85
reference score.

### Phase 7: Hard-Category Trace Policy, May 28 -> Kaggle 0.74

The next useful finding came from separating hard-category failures by cause:

| Category group | Observed issue | Tested response |
| --- | --- | --- |
| `bit_manipulation` | Labels could be mostly clean, but verbose traces were too long for reliable LoRA cloning | Compact reasoner |
| `cryptarithm_*` | Training labels were often wrong | Drop wrong hard-category traces |
| `equation_numeric_guess` | Low label correctness and little clean supervision | Drop wrong traces rather than teach bad boxed answers |

The compact `bit_manipulation` reasoner was the largest positive intervention:
it shortened verbose multi-thousand-token traces into a much more cloneable
column/rule-fitting format. The tradeoff was lower raw label correctness but
better learned behavior. In the recorded eval, `bit_manipulation` improved from
near-zero behavior to a materially useful category.

The hard-category drop policy was also a useful negative correction: keeping
wrong traces can teach procedure, but when a category's boxed answers are wrong
too often, the model learns wrong answer patterns. Smaller and cleaner was
better than larger and noisy for those categories.

Together these operations moved Kaggle to **0.74**, the best score reached.

### Phase 8: Expanded Target Modules / Capacity Test, May 29-31 -> Kaggle 0.74

The final tested hypothesis was that the remaining gap came from missing LoRA
capacity or missing module names. The config added candidate SwiGLU/MoE/Mamba
names such as `gate_proj`, `w1`, `w2`, `w3`, `gate_up_proj`, `dt_proj`, and
`x_proj`.

The operation increased trainable parameters from about 80M to roughly 884M and
produced a multi-GB adapter. The H100/vLLM probe showed the expanded adapter
shape could load. However, the score did not improve: Kaggle stayed at **0.74**,
and the local stratified eval moved downward in the recorded comparison
(`0.573 -> 0.558`).

This is best read as an ablation result: more trainable adapter capacity did
not help this data/config path, and it likely reduced useful regularization.
It is not a proof that capacity can never help, but it argues against simply
making this adapter larger without changing data quality or supervision.

---

## What Went Well

1. **A reproducible HPC/vLLM/LoRA path was built.** The project moved from
   "can we run the model?" to a chunked training and submission workflow.

2. **The THK reference reduced search waste.** Porting the reasoners,
   augmenters, suffix behavior, and training recipe made the project compare
   against a known strong route instead of drifting through arbitrary tuning.

3. **The local eval became more useful.** Moving from a single-category holdout
   to a 450-row stratified holdout gave signal on the categories that were
   actually limiting Kaggle performance.

4. **The best intervention was diagnostic, not just larger training.** The
   compact `bit_manipulation` trace was found by asking why clean labels still
   failed at eval time.

5. **The final capacity test was valuable even though it did not improve score.**
   It made the "just add more target modules" path less attractive.

---

## What Did Not Work

1. **Full-rank `lm_head` through `modules_to_save` did not survive the tested
   vLLM/Kaggle path.** This blocked a direct implementation of THK-style
   `train_unembed: true`.

2. **Early hyperparameter changes did not break the 0.53-0.57 plateau.** The
   same category profile kept reappearing until data and reasoner coverage
   changed.

3. **Verbose correct traces were not always learnable traces.** The original
   `bit_manipulation` traces were too long and brittle for the LoRA to clone
   into correct answers.

4. **"Procedure over correctness" was not universally safe.** It helped as a
   THK-style scaffolding idea, but it became harmful for low-solve-rate
   hard categories.

5. **The expanded 884M-parameter adapter did not improve the final score.** In
   this tested setting it looked more like overfitting than generalization.

---

## Ablation / Operation Summary

| Operation | Outcome | Interpretation |
| --- | --- | --- |
| First deterministic algo-CoT LoRA | Kaggle 0.53 | Basic submission path worked; coverage was incomplete |
| Several early training configs | Kaggle 0.53-0.57 | Recipe tuning was not the main bottleneck |
| Add equation/cryptarithm reasoners + stratified holdout + rank 32 | Kaggle 0.66 | Broader category coverage mattered |
| Try `modules_to_save=[lm_head]` | Kaggle/vLLM rejection | Full-rank replacement weights were not compatible with the tested serving path |
| Replace full-rank `lm_head` with rank-32 `lm_head` LoRA target | Restored vLLM-loadable route | Practical approximation of THK `train_unembed` |
| THK-style reasoners/augmenters/recipe | Kaggle about 0.68-0.70 | Reproduction helped but did not close the gap |
| Compact `bit_manipulation` | Major category improvement | Shorter cloneable traces beat verbose traces |
| Drop wrong hard-category traces | Net positive in final path | Prevented low-solve-rate categories from teaching wrong boxed answers |
| Expanded target modules to about 884M trainable params | Kaggle stayed 0.74, local eval regressed | More capacity did not help this data/config route |

---

## Current Interpretation

The fairest summary is:

> The project reached 0.74 under the tested Kaggle-compatible vLLM-LoRA path.
> The best evidence so far says the remaining gap is not solved by more
> ordinary hyperparameter tuning or by a much larger LoRA target-module set.
> The remaining plausible levers are better hard-category solvers, better clean
> supervision for guess-style categories, or an eval/training route that can
> use full-rank updates not accepted by the current public vLLM-LoRA contract.

That is intentionally narrower than saying the task is fully solved or making a
universal claim about the 0.85 reference. The project has tested a specific
operational route thoroughly enough to justify stopping or changing strategy,
but not enough to rule out every future solver or serving-stack change.

---

## Reader Notes

- Older repo notes and README sections are useful as a dated work log, but some
  early links and planning docs predate the final THK-style path. Treat this
  report, `configs/train_thk_full.yaml`, `docs/path_beta_runbook.md`,
  `scripts/build_sft_traces.py`, and `scripts/package_submission.py` as the key
  current-orientation files.
- Large artifacts are intentionally absent from git. This is expected. For
  visible reproducibility, attach small logs and config snapshots rather than
  multi-GB adapters.
- Kaggle score visibility is limited to the raw score, so per-category
  conclusions come from local stratified evals and controlled ablations.
