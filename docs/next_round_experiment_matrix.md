# Next-round experiment matrix

Goal: separate the next two global questions before changing the hard-category
policies themselves:

1. How much of the current gap comes from the train/eval prompt mismatch
   (`append_thk_suffix`)?
2. After prompt shape is aligned, how much extra gain comes from restoring the
   held-out 450 reasoner rows and the resulting extra `matching` examples?

This deliberately postpones policy changes for `bit_manipulation`,
`cryptarithm_*`, and `equation_numeric_guess` until those two broad effects are
measured cleanly.

## Matrix

| ID | Train prompt suffix | Reasoner corpus | Local stratified holdout valid? | Main question answered |
|---|---|---:|---|---|
| E0 | off | holdout-50 corpus (`17,391` combined rows) | yes | Historical baseline already measured by eval job `6857905` |
| E1 | on | holdout-50 corpus (`17,391` combined rows) | yes | Pure suffix effect |
| E2 | on | full-corpus rebuild (`17,963` combined rows expected) | no, because the 450 local holdout rows re-enter training | Added-data effect after suffix alignment |

## Why only these three

`E0 -> E1` isolates prompt-shape alignment while keeping the same holdout regime.
`E1 -> E2` isolates the value of recovering the withheld rows after the suffix
issue is removed. If we changed the hard-category policy at the same time, we
would no longer know which lever caused the gain.

## E1 — suffix-on, keep the current local holdout

Build:

```bash
python3 scripts/build_sft_traces.py \
  --keep-wrong-traces \
  --use-reasoner-boxed \
  --holdout 50

python3 scripts/build_augmenter_traces.py \
  --matching-source data/sft_traces.jsonl

python3 scripts/combine_traces.py
```

Train:

```bash
sbatch --export=ALL,THK_CHUNK_IDX=1 slurm/train_thk_chunk.slurm
sbatch --export=ALL,THK_CHUNK_IDX=2 slurm/train_thk_chunk.slurm
sbatch --export=ALL,THK_CHUNK_IDX=3 slurm/train_thk_chunk.slurm
```

Evaluate the final archived adapter from chunk 3:

```bash
LORA_ADAPTER_DIR=$HOME/Nemotron_Challenge/outputs/lora_adapter_thk_chunk3_<JOBID_3> \
  sbatch slurm/eval.slurm
```

Readout:

- Compare against `E0` on the same 450-row local stratified holdout.
- The cleanest signal is the change in:
  - overall balanced holdout accuracy
  - `bit_manipulation`
  - `cryptarithm_deduce`
  - `cryptarithm_guess`
  - `equation_numeric_guess`

## E2 — suffix-on, restore the full corpus

Build:

```bash
python3 scripts/build_sft_traces.py \
  --keep-wrong-traces \
  --use-reasoner-boxed \
  --holdout 0

python3 scripts/build_augmenter_traces.py \
  --matching-source data/sft_traces.jsonl

python3 scripts/combine_traces.py
```

Expected shape:

- `data/sft_traces.jsonl`: `9,500` reasoner rows
- `data/augmenter_traces.jsonl`: `8,463` rows expected if `matching`
  regains the 122 examples lost under the holdout-50 build
- `data/sft_combined.jsonl`: `17,963` rows expected

Train with the same 3-chunk route and submit to Kaggle.

Readout:

- Do **not** trust the old 450-row local holdout for this run; those prompts are
  now back inside training.
- The clean comparison is `E1 Kaggle score -> E2 Kaggle score`.

## Decision rule before hard-category policy work

- If `E1` moves materially above `E0`, keep the suffix permanently and do not
  mix unsuffixed data back in.
- If `E2` adds little after `E1`, the remaining ceiling is no longer a global
  recipe issue; move next to category-policy work.
- If `E1` barely moves, the prompt mismatch was real but not decisive; the next
  work should still be category-policy work rather than more THK cloning.

## E3AB - full corpus plus hard-category label repair

Use after E2 has confirmed the suffix-on, full-corpus path. This keeps the E2
row coverage and trace body policy, but rewrites wrong final boxed labels to
ground truth for the hard categories only:

- `bit_manipulation`
- `cryptarithm_deduce`
- `cryptarithm_guess`
- `equation_numeric_guess`

Build:

```bash
python3 scripts/build_sft_traces.py \
  --keep-wrong-traces \
  --use-reasoner-boxed \
  --e3ab-hard-label-repair \
  --holdout 0

python3 scripts/build_augmenter_traces.py \
  --matching-source data/sft_traces.jsonl

python3 scripts/combine_traces.py
```

Expected intent:

- `data/sft_traces.jsonl`: keep all `9,500` reasoner rows.
- Hard-category wrong traces are not dropped.
- Easy categories keep E2/THK `--use-reasoner-boxed` behavior.
- The final supervised `\boxed{...}` is corrected only for wrong hard-category
  rows.

Train, evaluate, and package with the same E2 chunked route. Treat this as a
score-seeking run: it intentionally combines corrected-label and hard-category
policy repair, so it is less diagnostic than separate A/B runs but cheaper in
GPU queue time.
