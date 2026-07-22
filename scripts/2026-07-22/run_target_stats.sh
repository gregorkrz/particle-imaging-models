#!/usr/bin/env bash
# Plot the LINEAR-PROBE config's per-particle training-target distributions
# (momentum, px/py/pz, vertex, is_primary -- as the heads see them) via
# `pimm run-stats`, into the s3-synced results dir, with a self-contained HTML.
#
# Runs inside the pimm apptainer image (CPU is fine). The pimm .env supplies
# PILARNET_DATA_ROOT_V3, so run-stats reads exactly the data the model trains on.
#
#   scripts/2026-07-22/run_target_stats.sh                 # val split, 1000 events
#   SPLIT=train NUM_EVENTS=5000 scripts/2026-07-22/run_target_stats.sh
#   EXTRA_ARGS="--drop-sentinel" scripts/2026-07-22/run_target_stats.sh
#   DRY_RUN=1 scripts/2026-07-22/run_target_stats.sh       # print the command only
#
# After it finishes, deploy with LArTPCFastSim/sync_results_to_remote.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# .env exports PILARNET_DATA_ROOT_V3 (the training data root) so the dataset
# resolves to the same files used in training. (It also holds a W&B key -- keep
# it sourced but never printed.)
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

# run-stats honors $OUTPUT_DIR; default to the s3-synced LArTPCFastSim results
# dir that sync_results_to_remote.sh pushes.
export OUTPUT_DIR="${OUTPUT_DIR:-/sdf/data/neutrino/gregork/lartpc_fastsim_outputs}"

CONFIG="${CONFIG:-${REPO_ROOT}/configs/panda/panseg/detector-v5-pt-v3m2-ft-joint-pxpypz-lin.py}"
SPLIT="${SPLIT:-val}"
NUM_EVENTS="${NUM_EVENTS:-1000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
IMAGE="${IMAGE:-/sdf/data/neutrino/youngsam/images/pimm_pytorch2.5.0-cuda12.4.sif}"

# `python -m pimm.run_stats` from the repo root uses THIS working copy of pimm
# (cwd shadows the image's installed pimm) so the new run-stats + lin config are
# picked up, with the container's torch / data stack.
INNER="export OUTPUT_DIR='${OUTPUT_DIR}' PILARNET_DATA_ROOT_V3='${PILARNET_DATA_ROOT_V3:-}'; \
cd '${REPO_ROOT}' && python -m pimm.run_stats '${CONFIG}' \
    --split '${SPLIT}' --num-events '${NUM_EVENTS}' ${EXTRA_ARGS}"

echo "Config     : ${CONFIG}"
echo "Split      : ${SPLIT}   events: ${NUM_EVENTS}   extra: ${EXTRA_ARGS:-(none)}"
echo "Data root  : ${PILARNET_DATA_ROOT_V3:-<unset>}"
echo "Output dir : ${OUTPUT_DIR}/run_stats/$(basename "${CONFIG%.py}")"
echo "Image      : ${IMAGE}"
echo "+ apptainer exec --nv --bind /sdf/home,/sdf/data/neutrino ${IMAGE} bash -c '<run-stats>'"

if [[ -z "${DRY_RUN:-}" ]]; then
    apptainer exec --nv --bind /sdf/home,/sdf/data/neutrino "${IMAGE}" \
        bash -c "${INNER}"
    echo
    echo "Done. Deploy with LArTPCFastSim/sync_results_to_remote.sh"
fi
