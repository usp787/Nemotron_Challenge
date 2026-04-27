# Data

Input prompt files for baseline and evaluation runs.

## Format

Each file is JSONL (one JSON object per line):

```json
{"id": "<unique_id>", "prompt": "<text>"}
```

Required fields:

- `id` — unique identifier per prompt
- `prompt` — raw text sent to the model

Optional fields can be added later (e.g. `category`, `reference_answer`)
without breaking the baseline reader.

## Files

- `sample_prompts_5.jsonl` — 5 hand-written prompts for quick smoke runs.
  Not tracked in git (`*.jsonl` is in `.gitignore`); regenerate locally
  or copy from another team member.
