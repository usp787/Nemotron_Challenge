"""Zip a trained LoRA adapter into submission.zip for Kaggle.

The Kaggle harness expects a flat zip whose root contains
``adapter_config.json`` and the adapter weights (typically
``adapter_model.safetensors``). We refuse to package if those files are
missing — better to fail locally than to silently submit a broken zip.

Tokenizer files are kept in the archive when present: harmless on the
Kaggle side and useful for local debug-loading via ``AutoPeftModel``.

Usage:
    python3 scripts/package_submission.py \\
        --adapter outputs/lora_adapter \\
        --output  outputs/submission.zip
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

REQUIRED = {"adapter_config.json"}
WEIGHT_NAMES = {"adapter_model.safetensors", "adapter_model.bin"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="Adapter directory (output of train_lora.py)")
    ap.add_argument("--output", default="outputs/submission.zip")
    args = ap.parse_args()

    adapter_dir = Path(args.adapter)
    if not adapter_dir.is_dir():
        raise SystemExit(f"Adapter directory not found: {adapter_dir}")

    files = {p.name for p in adapter_dir.iterdir() if p.is_file()}
    missing = REQUIRED - files
    if missing:
        raise SystemExit(f"Adapter missing required files: {sorted(missing)}")
    if not (files & WEIGHT_NAMES):
        raise SystemExit(
            f"No adapter weight file found in {adapter_dir}. "
            f"Expected one of: {sorted(WEIGHT_NAMES)}"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(adapter_dir.iterdir()):
            if p.is_file():
                zf.write(p, arcname=p.name)
                written.append(p.name)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path} ({size_mb:.2f} MB) containing {len(written)} files:")
    for name in written:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
