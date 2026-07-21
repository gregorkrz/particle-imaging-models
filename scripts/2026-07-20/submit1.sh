#!/bin/bash
# Launch command reconstructed from slurm job 32471877
# (detector-v5-pt-v3m2-ft-joint-pxpypz-fft, submitted 2026-07-20 15:16).
# Joint detector-v5 with px/py/pz momentum regression, WITH the random
# geometric augmentations (RandomRotate z/x/y + RandomFlip).
#
# Default: submit to S3DF Slurm (4 GPUs). Pass --gcloud to submit to Google
# Cloud Batch instead (single A100; the gcloud site supplies its own resources).
set -euo pipefail

CONFIG=panda/panseg/detector-v5-pt-v3m2-ft-joint-pxpypz-fft

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
