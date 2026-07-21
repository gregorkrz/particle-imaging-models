#!/bin/bash
# Same job as submit1.sh but WITHOUT the random geometric augmentations
# (no RandomRotate / RandomFlip in the train transform). Uses the ablation
# config detector-v5-pt-v3m2-ft-joint-pxpypz-fft-noaug, which inherits
# everything else from the augmented baseline.
#
# Default: submit to S3DF Slurm (4 GPUs). Pass --gcloud to submit to Google
# Cloud Batch instead (single A100; the gcloud site supplies its own resources).
set -euo pipefail

CONFIG=panda/panseg/detector-v5-pt-v3m2-ft-joint-pxpypz-fft-noaug

if [ "${1:-}" = "--gcloud" ]; then
  uv run pimm submit \
    --site gcloud \
    --resources.nnodes 1 \
    --train.config "$CONFIG"
else
  uv run pimm submit \
    --site s3df \
    --resources.nnodes 1 \
    --resources.nproc-per-node 4 \
    --resources.time 04:00:00 \
    --train.config "$CONFIG"
fi
