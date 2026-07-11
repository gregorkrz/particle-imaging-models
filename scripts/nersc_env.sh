#!/usr/bin/env bash
# Build MPI-enabled mpi4py and parallel h5py into the pimm environment, pinned
# to the versions in uv.lock.
#
# Two contexts share this script:
#   - a Perlmutter login node (bare-metal R&D environment): syncs the locked
#     environment into ./.venv, then rebuilds against the Cray toolchain,
#     following https://docs.nersc.gov/development/languages/python/parallel-python/
#   - the Dockerfile.nersc image build: rebuilds against the vanilla MPICH and
#     parallel HDF5 baked into the image (the sync happens in the Dockerfile)
#
# Run from the repository root. For multi-node bare-metal jobs, also
# `module load nccl` (matching your CUDA version) at run time.
set -euo pipefail

if command -v module >/dev/null 2>&1; then
  # Perlmutter bare metal: Cray compiler wrappers and system libraries
  module load PrgEnv-gnu cray-mpich cray-hdf5-parallel
  MPI4PY_MPICC="cc -shared"
  H5PY_CC="cc"
  uv sync --locked --group nersc
else
  # container image build: vanilla MPICH and /opt/hdf5 provided by the image
  MPI4PY_MPICC="mpicc"
  H5PY_CC="mpicc"
fi

MPI4PY_VERSION="$(uv run --no-sync python -c 'import importlib.metadata as m; print(m.version("mpi4py"))')"
H5PY_VERSION="$(uv run --no-sync python -c 'import importlib.metadata as m; print(m.version("h5py"))')"

MPICC="$MPI4PY_MPICC" uv pip install --reinstall --no-cache \
  --no-binary mpi4py "mpi4py==${MPI4PY_VERSION}"
HDF5_MPI=ON CC="$H5PY_CC" uv pip install --reinstall --no-cache \
  --no-binary h5py "h5py==${H5PY_VERSION}"

uv run --no-sync python -c "import h5py, mpi4py; assert h5py.get_config().mpi, 'h5py built without MPI'"
echo "MPI-enabled environment ready"
