#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WITH_FLASH=1
LAUNCHER_ONLY=0

usage() {
  cat <<'EOF'
Usage: ./install.sh [--no-flash | --launcher-only]

  --no-flash       install the GPU training environment without FlashAttention
  --launcher-only  install only the pimm launch/submit command dependencies
EOF
}

fail() {
  echo "error: $*" >&2
  exit 1
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi
  command -v curl >/dev/null 2>&1 || fail "curl is required to install uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || fail "uv was installed but is not on PATH"
}

cuda_version() {
  nvcc --version | awk '
    /release/ {
      for (i = 1; i <= NF; i++) {
        if ($i == "release") {
          gsub(/,/, "", $(i + 1))
          print $(i + 1)
          exit
        }
      }
    }
  '
}

detect_cuda_home() {
  if [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
    return
  fi
  command -v nvcc >/dev/null 2>&1 || fail \
    "CUDA 12.4 with nvcc is required; install it or load your site's CUDA module"
  local nvcc_path
  nvcc_path="$(readlink -f "$(command -v nvcc)")"
  export CUDA_HOME="$(dirname "$(dirname "$nvcc_path")")"
}

detect_arches() {
  if [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    return
  fi
  command -v nvidia-smi >/dev/null 2>&1 || fail \
    "nvidia-smi is required to detect GPU architectures; set TORCH_CUDA_ARCH_LIST to build without it"
  local capabilities
  capabilities="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)"
  [ -n "$capabilities" ] || fail \
    "could not detect GPU compute capability; set TORCH_CUDA_ARCH_LIST, for example 8.0 or 9.0"
  export TORCH_CUDA_ARCH_LIST
  TORCH_CUDA_ARCH_LIST="$(printf '%s\n' "$capabilities" | tr -d ' ' | sort -u | paste -sd' ' -)"
}

validate_toolchain() {
  [ "$(uname -s)" = "Linux" ] || fail "GPU training installs support Linux only"
  [ "$(uname -m)" = "x86_64" ] || fail "GPU training installs support x86_64 only"

  detect_cuda_home
  export PATH="$CUDA_HOME/bin:$PATH"
  [ "$(cuda_version)" = "12.4" ] || fail \
    "CUDA 12.4 is required, but nvcc reports $(cuda_version || echo unknown)"

  export CC="${CC:-gcc}"
  export CXX="${CXX:-g++}"
  command -v "$CC" >/dev/null 2>&1 || fail "a GCC host compiler is required"
  command -v "$CXX" >/dev/null 2>&1 || fail "a G++ host compiler is required"
  local cxx_major
  cxx_major="$("$CXX" -dumpfullversion -dumpversion | cut -d. -f1)"
  [ "$cxx_major" -ge 9 ] && [ "$cxx_major" -le 12 ] || fail \
    "G++ 9 through 12 is required for the CUDA 12.4 extension builds"

  printf '#include <google/sparse_hash_map>\n' | "$CXX" -E -x c++ - >/dev/null 2>&1 || fail \
    "sparsehash headers are required; install libsparsehash-dev or load an equivalent module"

  detect_arches
  export FORCE_CUDA=1
  export MAX_JOBS="${MAX_JOBS:-2}"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --no-flash)
      WITH_FLASH=0
      ;;
    --launcher-only)
      LAUNCHER_ONLY=1
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      fail "unknown option: $1"
      ;;
  esac
  shift
done

cd "$HERE"
ensure_uv

if [ "$LAUNCHER_ONLY" -eq 1 ]; then
  uv sync --locked
  echo "launcher environment ready. run commands with: uv run pimm"
  exit 0
fi

validate_toolchain
echo "CUDA_HOME=$CUDA_HOME"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

if [ "$WITH_FLASH" -eq 1 ]; then
  uv sync --all-extras --locked
else
  uv sync --extra train --locked
fi

uv run --no-sync python - <<PY
import importlib

modules = [
    "cnms",
    "pointgroup_ops",
    "pointops",
    "pointrope",
    "pytorch3d_ops",
    "serialize_cuda",
    "spconv",
    "torch",
    "torch_cluster",
    "torch_scatter",
    "torch_sparse",
]
if $WITH_FLASH:
    modules.append("flash_attn")
for module in modules:
    importlib.import_module(module)
print("validated:", ", ".join(modules))
PY

echo "training environment ready. run commands with: uv run pimm"
