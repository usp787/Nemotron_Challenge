"""Download AIME 2025 and write data/aime25.jsonl.

Source: MathArena/aime_2025 on Hugging Face. This is the most widely
cited public AIME 2025 mirror (used by MathArena's leaderboard, the HF
evaluation guidebook, and several recent reasoning-model papers).
NVIDIA's own evaluation in the Nemotron 3 Nano report does not disclose
which mirror it used, so this is a faithful but not bit-identical
substitute.

We pull rows via the public Hugging Face datasets-server REST API
instead of the `datasets` library, because the vLLM container does not
ship `datasets` and we don't want to inject a `pip install --user`
into the run (the Slurm scripts explicitly set PYTHONNOUSERSITE=1).
This keeps the prep step pure-stdlib and runnable from any Python 3
on a node with outbound internet (the login node).

Each output record has:
  - id (str): "aime25_<two-digit problem index>"
  - prompt (str): the NeMo-Skills boxed-answer math template applied to
    the problem statement. We use this template because NVIDIA's exact
    prompt config (math-oai.yaml) ships only inside the proprietary
    eval-factory container; the public NeMo-Skills generic/math.yaml is
    the closest published equivalent and uses the same convention.
  - expected_answer (int): ground-truth integer answer (0-999)
  - problem_idx (int): original index in the competition
  - problem_type (list[str]): topic tags from MathArena

Usage:
    python3 scripts/prepare_aime25.py
"""
import json
import urllib.request
from pathlib import Path


ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=MathArena%2Faime_2025"
    "&config=default&split=train&offset=0&length=100"
)

PROMPT_TEMPLATE = (
    "Solve the following math problem. Make sure to put the answer "
    "(and only answer) inside \\boxed{{}}.\n\n{problem}"
)


def fetch_rows() -> list[dict]:
    with urllib.request.urlopen(ROWS_URL, timeout=60) as resp:
        payload = json.load(resp)
    return [entry["row"] for entry in payload["rows"]]


def main() -> None:
    rows = fetch_rows()
    out_path = Path("data/aime25.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            rec = {
                "id": f"aime25_{int(row['problem_idx']):02d}",
                "prompt": PROMPT_TEMPLATE.format(problem=row["problem"]),
                "expected_answer": int(row["answer"]),
                "problem_idx": int(row["problem_idx"]),
                "problem_type": list(row.get("problem_type") or []),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} prompts -> {out_path}")


if __name__ == "__main__":
    main()
