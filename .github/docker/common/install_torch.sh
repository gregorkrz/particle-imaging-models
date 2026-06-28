#!/usr/bin/env bash
# Install torch + torchvision at the pinned version + CUDA, from the official index.
# (torchaudio is not used by pimm.) The Docker images start FROM a pytorch base
# that already ships torch, so the Dockerfiles SKIP this; it is for install.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/versions.sh"

echo ">>>>>>>>> installing torch ${TORCH_VERSION} (cu${CUDA_VERSION_NO_DOT}) >>>>>>>>>"
pip install --no-cache-dir \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION_NO_DOT}"
