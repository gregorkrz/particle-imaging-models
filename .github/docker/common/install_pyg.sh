#!/usr/bin/env bash
# CUDA wheels pinned to the torch+CUDA build: torch_scatter/sparse/cluster (from
# the PyG find-links index) and spconv (cuda-suffixed wheel). Requires torch.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/versions.sh"
REQ="${PIMM_REQUIREMENTS:-$HERE/../requirements}"

echo ">>>>>>>>> installing PyG wheels + spconv for torch ${TORCH_VERSION}+cu${CUDA_VERSION_NO_DOT} >>>>>>>>>"
pip install --no-cache-dir -r "$REQ/requirements-cuda.txt" \
  -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION_NO_DOT}.html"
pip install --no-cache-dir "spconv-cu${CUDA_VERSION_NO_DOT}"
