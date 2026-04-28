"""Download AIME 2025 and write data/aime25.jsonl.

Source: MathArena/aime_2025 on Hugging Face. This is the most widely
cited public AIME 2025 mirror (used by MathArena's leaderboard, the HF
evaluation guidebook, and several recent reasoning-model papers).
NVIDIA's own evaluation in the Nemotron 3 Nano report does not disclose
which mirror it used, so this is a faithful but not bit-identical
substitute.

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

Run this once on the cluster login node (where outbound internet
works) before submitting the baseline Slurm job. The compute node does
not need internet access during inference.

Usage:
    python3 scripts/prepare_aime25.py
"""
import json
from pathlib import Path

from datasets import load_dataset


PROMPT_TEMPLATE = (
    "Solve the following math problem. Make sure to put the answer "
    "(and only answer) inside \\boxed{{}}.\n\n{problem}"
)


def main() -> None:
    ds = load_dataset("MathArena/aime_2025", split="train")
    out_path = Path("data/aime25.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            rec = {
                "id": f"aime25_{int(row['problem_idx']):02d}",
                "prompt": PROMPT_TEMPLATE.format(problem=row["problem"]),
                "expected_answer": int(row["answer"]),
                "problem_idx": int(row["problem_idx"]),
                "problem_type": list(row.get("problem_type") or []),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(ds)} prompts -> {out_path}")


if __name__ == "__main__":
    main()
