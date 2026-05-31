# Data

Input prompt files for baseline and evaluation runs.

## Format

Each file is JSONL (one JSON object per line):

```json
{"id": "<unique_id>", "prompt": "<text>"}
```

Required fields:

- `id` - unique identifier per prompt
- `prompt` - raw text sent to the model

Optional fields can be added later (e.g. `category`, `reference_answer`)
without breaking the baseline reader.

## Files

- `sample_prompts_5.jsonl` - 5 hand-written prompts for quick smoke runs.
  This small smoke-test fixture is tracked in git so a fresh cluster clone can
  run Stage 6 without manual regeneration.

## THK-style training snapshot

The tracked repo snapshot also includes the small-enough JSONL artifacts used by
the THK-style LoRA pipeline:

| File | Purpose | Tracked row count |
| --- | --- | ---: |
| `problems_classified.jsonl` | 9-category classified Kaggle training prompts | 9,500 |
| `sft_traces.jsonl` | Reasoner SFT traces after the holdout split | 9,050 |
| `augmenter_traces.jsonl` | THK-style augmenter traces | 8,341 |
| `sft_combined.jsonl` | Combined reasoner + augmenter training corpus | 17,391 |
| `stratified_holdout.jsonl` | 50 prompts per category for local eval | 450 |

Large eval outputs, adapters, and final cluster logs are intentionally not
tracked here. See `Project_Report.md` for the retrospective and the expected
small evidence artifacts to attach manually.
