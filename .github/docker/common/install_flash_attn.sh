#!/usr/bin/env bash
# FlashAttention 2 (built from source; slow). Optional - configs run without it
# when enable_flash=False. Requires torch.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/versions.sh"

echo ">>>>>>>>> installing flash-attn ${FLASH_ATTN_VERSION} (from source, slow) >>>>>>>>>"
MAX_JOBS="${MAX_JOBS:-1}" pip install --no-cache-dir \
  "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation
