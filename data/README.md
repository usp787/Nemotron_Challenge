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
