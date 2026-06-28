#!/usr/bin/env bash
# Create a conda env for pimm that matches the Docker image, reusing the same
# pins (.github/docker/common/versions.sh) and install scripts (.github/docker/common/*).
#
#   ./install.sh                 # GPU env, builds flash-attn (matches the image)
#   ./install.sh --no-flash      # skip flash-attn (configs run with enable_flash=False)
#   ./install.sh --cpu           # CPU-only: skip CUDA wheels/extensions/flash
#   ./install.sh --name myenv    # custom env name
#
# After it finishes:  conda activate <env>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/.github/docker/common/versions.sh"

ENV_NAME="pimm-torch${TORCH_VERSION}-cu${CUDA_VERSION_NO_DOT}"
WITH_FLASH=1
CPU_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --name) ENV_NAME="$2"; shift 2 ;;
    --flash) WITH_FLASH=1; shift ;;
    --no-flash) WITH_FLASH=0; shift ;;
    --cpu) CPU_ONLY=1; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

command -v conda >/dev/null || { echo "conda not found on PATH" >&2; exit 1; }
# Use mamba for the (slow) solves if available; fall back to conda.
SOLVER=conda
command -v mamba >/dev/null 2>&1 && SOLVER=mamba

echo "creating env: $ENV_NAME (python ${PYTHON_VERSION}) via ${SOLVER}"
"$SOLVER" create -y -n "$ENV_NAME" -c conda-forge "python=${PYTHON_VERSION}" cmake ninja
if [ "$CPU_ONLY" -eq 0 ]; then
  # Toolchain for building the CUDA extensions: nvcc + a host compiler it accepts
  # (conda-forge's cuda-toolkit 12.4 nvcc requires gcc < 13) + sparsehash headers.
  "$SOLVER" install -y -n "$ENV_NAME" -c conda-forge \
    "cuda-toolkit=${CUDA_VERSION}.*" "gcc=12.*" "gxx=12.*" sparsehash
fi

# conda's activation scripts (e.g. activate-gcc_linux-64.sh referencing
# SYS_SYSROOT) are not `set -u`-clean, so relax nounset around activation.
set +u
# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
set -u

# Build the CUDA extensions against the env's conda toolkit, not a system CUDA the
# shell may export (e.g. `module load cuda` or /usr/local/cuda).
export CUDA_HOME="$CONDA_PREFIX"
unset CUDA_PATH

export PIMM_REQUIREMENTS="$HERE/.github/docker/requirements"
export PIMM_LIBS="$HERE/libs"

bash "$HERE/.github/docker/common/install_torch.sh"
bash "$HERE/.github/docker/common/install_python_deps.sh"
if [ "$CPU_ONLY" -eq 0 ]; then
  bash "$HERE/.github/docker/common/install_pyg.sh"
  [ "$WITH_FLASH" -eq 1 ] && bash "$HERE/.github/docker/common/install_flash_attn.sh"
  bash "$HERE/.github/docker/common/install_cuda_exts.sh"
fi
pip install --no-cache-dir -e "$HERE"

echo
echo "done. activate with:  conda activate $ENV_NAME"
