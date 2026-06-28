#!/usr/bin/env bash
# Build the local CUDA extensions under libs/. Needs torch, a CUDA toolchain
# (nvcc + g++), and TORCH_CUDA_ARCH_LIST. Point PIMM_LIBS at the libs/ dir.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/versions.sh"
LIBS="${PIMM_LIBS:-$HERE/../../libs}"

echo ">>>>>>>>> building CUDA extensions from ${LIBS} (TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}) >>>>>>>>>"
export TORCH_CUDA_ARCH_LIST MAX_JOBS="${MAX_JOBS:-1}"
pip install --no-cache-dir "$LIBS/pointrope" -v --no-build-isolation
pip install --no-cache-dir "$LIBS/pointops" -v --no-build-isolation
pip install --no-cache-dir "$LIBS/pointgroup_ops" -v --no-build-isolation
pip install --no-cache-dir "$LIBS/cnms" -v --no-build-isolation
FORCE_CUDA=1 pip install --no-cache-dir "$LIBS/pytorch3d_ops" -v --no-build-isolation
