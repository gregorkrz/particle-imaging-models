#!/usr/bin/env bash
# Plain pip dependencies (no torch, no CUDA wheels, no flash-attn, no local libs).
# Requires torch already installed (torch-geometric / ocnn / CLIP need it).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/versions.sh"
REQ="${PIMM_REQUIREMENTS:-$HERE/../requirements}"

echo ">>>>>>>>> installing python deps from ${REQ}/requirements.txt >>>>>>>>>"
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -r "$REQ/requirements.txt"
