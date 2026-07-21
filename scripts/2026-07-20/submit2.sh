#!/bin/bash
# Same job as submit1.sh but WITHOUT the random geometric augmentations
# (no RandomRotate / RandomFlip in the train transform). Uses the ablation
# config detector-v5-pt-v3m2-ft-joint-pxpypz-fft-noaug, which inherits
# everything else from the augmented baseline.
set -euo pipefail

uv run pimm submit \
  --site s3df \
  --resources.nnodes 1 \
  --resources.nproc-per-node 4 \
  --resources.time 04:00:00 \
  --train.config panda/panseg/detector-v5-pt-v3m2-ft-joint-pxpypz-fft-noaug
