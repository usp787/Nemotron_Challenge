#!/bin/bash
#SBATCH --job-name=pull_vllm_lora_eval
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Pull the newer vLLM image used only for LoRA-enabled evaluation.
#
# No GPU is needed for this job. Run it on a CPU/short allocation to avoid
# burning scarce H200 time on image conversion.

set -euo pipefail

IMAGE_TAG=${VLLM_LORA_EVAL_IMAGE:-vllm/vllm-openai:v0.20.0-cu129}
CONTAINER=${VLLM_LORA_EVAL_CONTAINER:-$SCRATCH/containers/nemotron_vllm_v0200_cu129.sif}

mkdir -p "$SCRATCH/containers" logs

echo "Image: docker://$IMAGE_TAG"
echo "Output: $CONTAINER"

if [ -f "$CONTAINER" ]; then
    echo "Container already exists; leaving it unchanged."
    ls -lh "$CONTAINER"
else
    apptainer pull "$CONTAINER" "docker://$IMAGE_TAG"
fi

echo ""
echo "=== Container Python/vLLM check ==="
apptainer exec "$CONTAINER" bash -lc "
  python3 --version
  python3 -c 'import vllm; print(\"vllm\", getattr(vllm, \"__version__\", \"unknown\"))'
"

echo ""
echo "Finished at: $(date)"
