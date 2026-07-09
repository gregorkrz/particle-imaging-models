# uv migration cluster handoff

Last updated: 2026-07-09 at approximately 13:22 Pacific time.

This file records the current state of the migration from Conda and scattered pip installation scripts to a single uv-managed project.
It is intended to make a new Cursor session on a Linux GPU cluster immediately productive.

## Repository state

- Local repository: `/Users/youngsam/code/pimm-site/particle-imaging-models`
- Branch: `docs/ux-audit-fixes`
- Starting commit: `52e63ab`
- The migration is entirely uncommitted.
- The branch already contained a large documentation rewrite before this migration began.
- Preserve every existing modification, deletion, and untracked file.
- Do not reset, restore, clean, or switch branches.
- Do not edit `/Users/youngsam/.cursor/plans/uv_installation_migration_7da67e9a.plan.md`.
- No commit has been requested or created.

The latest `git status --short` should be captured again immediately after moving the workspace.
The most important untracked migration files are:

```text
.python-version
uv.lock
libs/cnms/pyproject.toml
libs/pointgroup_ops/pyproject.toml
libs/pointops/pyproject.toml
libs/pointrope/pyproject.toml
libs/pytorch3d_ops/pyproject.toml
libs/serialize_cuda/pyproject.toml
UV_MIGRATION_CLUSTER_HANDOFF.md
```

There are unrelated or earlier documentation files in the same working tree, including many modified docs pages, deleted audit files, `docs/source/_static/custom-icons.js`, and generated docs directories.
Do not infer that every dirty file belongs to the uv migration.

Before doing anything on the cluster, confirm:

```bash
git branch --show-current
git rev-parse --short HEAD
git status --short
git diff --check
```

The expected branch is `docs/ux-audit-fixes`, and the expected base revision is `52e63ab`.

## Goal and decisions

The goal is to replace the local Conda installation with uv while using the same locked dependency set for local training, Docker, NERSC, and documentation CI.

Two decisions were made explicitly:

1. Local GPU installation requires a host-provided CUDA 12.4 compiler toolchain.
2. `uv.lock` is the dependency source for local installation, Docker images, and docs CI.

The local installer does not install CUDA, GCC, G++, or sparsehash.
It validates those system components and provides actionable errors.

The supported training platform is currently Linux x86_64 with Python 3.10, CUDA 12.4, and PyTorch 2.5.0.
The launcher-only profile also supports Apple Silicon macOS.

An A100 should use:

```bash
export TORCH_CUDA_ARCH_LIST=8.0
```

Using only `8.0` on the cluster will make extension compilation much faster than the multi-architecture Docker default.

## Todo state

The implementation todo state when this handoff was written was:

- Completed: consolidate dependencies, indexes, groups, Python pin, and lockfile in the uv project.
- Completed: add uv-aware build metadata and cache invalidation for local CUDA extensions.
- Completed: replace the Conda bootstrap with host-toolchain validation and uv sync.
- Completed: migrate Docker, NERSC, docs CI, and lock checks to uv.
- Completed: replace Conda and pip instructions with canonical uv workflows.
- In progress: validate clean, repeated, no-FlashAttention, launcher-only, standard image, NERSC MPI, and docs paths.

The remaining work is validation and any fixes revealed by native Linux execution.

## Implemented dependency model

The root `pyproject.toml` now contains the dependency model.

The base project dependencies are intentionally small:

```text
PyYAML
submitit
tyro
addict
yapf
```

The `train` extra contains the Linux GPU training stack.
It includes PyTorch 2.5.0, torchvision 0.20.0, PyG packages, spconv for CUDA 12.4, training utilities, and all six local CUDA packages.

The `flash-attn` extra uses the prebuilt wheel:

```text
flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

The direct wheel avoids rebuilding FlashAttention from source.

Dependency groups are:

- `dev` for Black, JupyterLab, and pytest.
- `docs` for Sphinx and theme extensions.
- `nersc` for Cython and mpi4py.

The project pins Python through:

```toml
requires-python = ">=3.10,<3.11"
```

The `.python-version` file contains:

```text
3.10
```

The uv project configuration currently includes:

```toml
[tool.uv]
required-version = ">=0.11.1"
default-groups = []
concurrent-builds = 1
```

Serial source builds are intentional.
Compiling multiple PyTorch CUDA extensions concurrently exhausted Docker Desktop memory even when each extension used `MAX_JOBS=1`.

The PyTorch index is:

```text
https://download.pytorch.org/whl/cu124
```

The PyG wheel index is:

```text
https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

The committed lock currently resolves 225 packages across Linux x86_64 and Apple Silicon macOS.

## Local CUDA packages

The six uv-managed local packages are:

```text
cnms
pointgroup-ops
pointops
pointrope
pytorch3d-ops
serialize-cuda
```

Each package now has a `pyproject.toml` with static name, version, and `torch` dependency metadata.
Static metadata is required for uv's `match-runtime = true` build dependency support.

Each package also defines uv cache keys for:

```text
pyproject.toml
setup.py
Python sources
C++ sources
CUDA sources
headers
TORCH_CUDA_ARCH_LIST
CUDA_HOME
CC
CXX
```

The root project supplies matching runtime PyTorch plus Ninja as extra build dependencies for each package.

## Local installer

`install.sh` has been rewritten.

Supported commands are:

```bash
./install.sh
./install.sh --no-flash
./install.sh --launcher-only
./install.sh --help
```

The full installer:

- installs uv if it is missing;
- requires Linux x86_64;
- locates CUDA through `CUDA_HOME` or `nvcc`;
- requires CUDA 12.4;
- requires GCC and G++ major version 9 through 12;
- verifies `google/sparse_hash_map` can be preprocessed;
- detects GPU compute capabilities through `nvidia-smi` unless `TORCH_CUDA_ARCH_LIST` is set;
- sets `FORCE_CUDA=1`;
- sets `MAX_JOBS=2` unless overridden;
- runs `uv sync --all-extras --locked`;
- imports the training stack after installation.

The no-FlashAttention path runs:

```bash
uv sync --extra train --locked
```

The launcher-only path runs:

```bash
uv sync --locked
```

## Docker design at handoff time

Both Dockerfiles copy uv 0.11.28 from `ghcr.io/astral-sh/uv`.
The uv environment is stored at `/opt/pimm/.venv`, outside the source checkout.
The environment is placed first on `PATH`.

The standard Dockerfile currently performs one complete locked sync:

```dockerfile
COPY pyproject.toml uv.lock .python-version ./
COPY libs/ ./libs/
COPY pimm/ ./pimm/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --all-extras --group docs --locked
```

It then copies non-package assets while excluding the dependency inputs and `pimm/`.
This keeps the final environment synchronized and prevents docs or configuration edits from invalidating native extension compilation.

The Docker build sets:

```text
CUDA_HOME=/usr/local/cuda
FORCE_CUDA=1
MAX_JOBS=1
UV_LINK_MODE=copy
UV_PROJECT_ENVIRONMENT=/opt/pimm/.venv
```

The default image architecture list remains:

```text
8.0 8.6 8.9 9.0
```

Override it with `8.0` for an A100-only validation build.

The NERSC Dockerfile uses the same uv sync pattern, adds MPICH 4.0.2 and parallel HDF5 1.14.6, then rebuilds the locked mpi4py and h5py versions against those libraries.
It asserts that `h5py.get_config().mpi` is true.

The NERSC and S3DF launcher interpreter paths now point to:

```text
/opt/pimm/.venv/bin/python
```

### Docker cache issue that was resolved

An earlier Docker design ran `uv sync --no-install-project`, copied the remaining source, and ran a second sync.
The second sync rebuilt every local CUDA package because the source copy invalidated local package state.

A subsequent design installed only the root package with `uv pip install`.
That produced working imports, but a later `uv sync` considered the project state incomplete and revisited all local packages.

The current design includes `pimm/` in the single locked sync.
This is intended to leave the final image fully synchronized.

One remaining performance tradeoff should be evaluated after correctness is proven.
A change under `pimm/` invalidates the locked sync Docker layer, so a fresh image build may revisit local CUDA packages.
Changes outside `pimm/`, `libs/`, `pyproject.toml`, and `uv.lock` should not invalidate that layer.

Do not optimize this by returning to the known-bad two-sync design.
If pimm-only image rebuilds remain too expensive, consider a dedicated extension wheel stage or a tested full-state bootstrap that preserves uv synchronization.

## CI changes

`.github/workflows/docker.yml` now:

- rebuilds for `.python-version`, `pyproject.toml`, and `uv.lock` changes;
- validates the lock with `uv lock --check`;
- makes both image jobs depend on the lock check;
- removes the obsolete `CUDA_VERSION_NO_DOT` argument.

`.github/workflows/docs.yml` now builds with the uv environment inside the project image.
It mounts the current checkout and sets:

```text
PYTHONPATH=/work
```

That setting is required so registry generation and autodoc import the checked-out source instead of the source baked into the image.

The CI command is:

```bash
docker run --rm -v "$PWD":/work -w /work -e PYTHONPATH=/work \
  youngsm/pimm:pytorch2.5.0-cuda12.4 \
  bash -lc "uv run --project /opt/pimm/src --no-sync make -C docs html"
```

There is a deployment-order risk to resolve before merging.
On the first main-branch commit containing this migration, the docs workflow could pull the previous published image before the Docker workflow publishes the uv-enabled image.
Possible resolutions include publishing the image first, combining workflow dependencies, or making docs CI build or select an image tied to the current revision.

## Deleted installation files

The following old installation machinery was removed:

```text
.github/docker/common/install_cuda_exts.sh
.github/docker/common/install_flash_attn.sh
.github/docker/common/install_pyg.sh
.github/docker/common/install_python_deps.sh
.github/docker/common/install_torch.sh
.github/docker/common/versions.sh
.github/docker/requirements/requirements-cuda.txt
.github/docker/requirements/requirements-dev.txt
.github/docker/requirements/requirements.txt
docs/requirements.txt
```

The remaining `install_system.sh` handles only system packages.

## Documentation changes

The installation documentation now describes:

- uv installation;
- host CUDA 12.4 prerequisites;
- compiler and sparsehash requirements;
- the prebuilt FlashAttention wheel;
- repeated install behavior;
- no-FlashAttention installation;
- launcher-only installation;
- Docker and Apptainer usage;
- environment verification;
- common CUDA and compiler failures.

Commands in the main README, docs landing page, quickstart, and HPC guide were updated to use `uv run` where commands execute from a local checkout.

Some remaining mentions of Conda are intentional.
The Docker entrypoint neutralizes Conda and Mamba initialization blocks from bind-mounted shell files because the base image still contains Conda.

## Bugs found during Docker validation

### Absolute PyTorch3D source paths

The first native wheel build failed because `libs/pytorch3d_ops/setup.py` passed absolute source paths to setuptools.
PEP 517 wheel builds require source paths relative to `setup.py`.

The fix keeps source paths relative while retaining an absolute compiler include directory:

```python
this_dir = os.path.dirname(os.path.abspath(__file__))
extensions_dir = os.path.join("pytorch3d_ops", "csrc")
include_dirs = [os.path.join(this_dir, extensions_dir)]
```

Do not revert the include directory to a relative path.
A relative include directory caused compilation to fail with:

```text
fatal error: utils/pytorch3d_cutils.h: No such file or directory
```

The PyTorch3D package `pyproject.toml` also now includes static description, author, and homepage metadata.

### Concurrent extension builds

uv initially built all six source distributions concurrently.
Docker ran out of memory while compiling `cnms`, even though Ninja used one worker.

The root uv configuration now sets:

```toml
concurrent-builds = 1
```

The Dockerfiles also set:

```text
MAX_JOBS=1
```

This made compilation reliable under Docker Desktop emulation, although very slow.

### Duplicate Docker extension builds

The earlier two-sync Docker design rebuilt native packages in its second sync.
The current Dockerfile uses one complete sync before copying non-package assets.

## Validation completed so far

The following checks passed:

- `uv lock --check`
- `git diff --check`
- Bash syntax checks for the installer and remaining shell helpers
- TOML parsing for the root and extension package metadata
- YAML parsing for both workflows and both site files
- IDE lint checks for edited Python files
- launcher-only installation on Apple Silicon macOS
- repeated launcher-only installation
- `uv run pimm --help`
- launcher dry-run for `panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask`
- locked Linux resolution for all extras and docs
- locked Linux resolution for the no-FlashAttention profile
- locked macOS resolution for the launcher-only profile
- installer help output
- rejection of a full GPU install on unsupported macOS
- standard Docker package imports using a successfully built intermediate image design
- documentation build inside that standard image
- repeat Docker build cache behavior for that intermediate image design

The standard image import validation successfully imported:

```text
cnms
flash_attn
pointgroup_ops
pointops
pointrope
pytorch3d_ops
serialize_cuda
spconv
torch
torch_cluster
torch_scatter
torch_sparse
```

The interpreter reported by that image was:

```text
/opt/pimm/.venv/bin/python
```

The docs build completed successfully in approximately 77 seconds under amd64 emulation.

## Validation not yet completed

The following checks still need native cluster validation:

- clean full local install with CUDA 12.4;
- actual A100 execution of custom CUDA kernels;
- repeated full install with proof that no extension rebuild occurs;
- actual no-FlashAttention transition and restore;
- separate launcher-only environment on Linux;
- final standard Dockerfile build after the single-sync redesign;
- final image `uv sync --check`;
- final image profile switch without native rebuilds;
- final docs build using the latest image design;
- NERSC image build;
- mpi4py and parallel h5py runtime checks;
- selected CUDA tests;
- GitHub workflow linting.

At handoff time, a local standard image build using the latest Dockerfile was still running under amd64 emulation.
Its build identifier is not useful outside the original Cursor session.
The last observed phase was the single locked uv sync compiling the six local extensions.
Do not assume that build completed successfully.

## Cluster prerequisites

Run validation on a Linux x86_64 compute node with an A100 assigned to the job.
Do not run GPU checks on a login node.

The required tools and headers are:

- CUDA toolkit 12.4 with `nvcc`;
- NVIDIA driver compatible with CUDA 12.4;
- GCC and G++ major version 9 through 12;
- `google/sparse_hash_map`;
- Git;
- curl if uv is not already installed;
- sufficient temporary storage for PyTorch, CUDA wheels, and extension builds.

Before installation, collect:

```bash
uname -s
uname -m
nvidia-smi
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
command -v nvcc
nvcc --version
command -v gcc
gcc --version
command -v g++
g++ --version
df -h .
```

Set or verify:

```bash
export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
export PATH="$CUDA_HOME/bin:$PATH"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=2
```

Verify sparsehash:

```bash
printf '#include <google/sparse_hash_map>\n' | "$CXX" -E -x c++ - >/dev/null
```

If the cluster provides CUDA or compilers through modules, load the site-specific CUDA 12.4 and GCC 9-12 modules before running the installer.
If sparsehash is installed in a nonstandard prefix, add its `include` directory to `CPLUS_INCLUDE_PATH`.

## Cluster validation runbook

### 1. Preserve and inspect the workspace

Confirm that the transferred workspace includes all tracked modifications and untracked migration files.
Pay particular attention to `uv.lock`, `.python-version`, the six extension `pyproject.toml` files, and this handoff.

Run:

```bash
git status --short
git diff --check
uv lock --check
```

Do not regenerate the lock unless `uv lock --check` demonstrates that it is stale and the reason is understood.

### 2. Validate configuration syntax

Run:

```bash
bash -n install.sh .github/docker/common/install_system.sh .github/docker/entrypoint.sh
```

Run the TOML and YAML parser check:

```bash
uv run --with tomli python - <<'PY'
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import yaml

for path in [Path("pyproject.toml"), *Path("libs").glob("*/pyproject.toml")]:
    with path.open("rb") as handle:
        tomllib.load(handle)

for path in [
    Path(".github/workflows/docker.yml"),
    Path(".github/workflows/docs.yml"),
    Path("launch/sites/nersc.yaml"),
    Path("launch/sites/s3df.yaml"),
]:
    with path.open() as handle:
        yaml.safe_load(handle)

print("configuration syntax OK")
PY
```

### 3. Create an isolated full training environment

Avoid deleting an existing environment.
Use a dedicated environment path for cluster validation:

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-cluster-full"
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=2
time ./install.sh 2>&1 | tee /tmp/pimm-uv-first-install.log
```

Expected behavior:

- uv resolves only locked versions;
- FlashAttention is installed from the prebuilt wheel;
- each of the six local packages is built once;
- the installer imports all required modules;
- no CUDA kernel is executed by the import check.

If the compute node has limited memory, set:

```bash
export MAX_JOBS=1
```

The root uv configuration already limits package builds to one at a time.

### 4. Verify the full environment and A100

Run:

```bash
uv run --no-sync python - <<'PY'
import importlib
import sys

import torch

modules = [
    "cnms",
    "flash_attn",
    "pointgroup_ops",
    "pointops",
    "pointrope",
    "pytorch3d_ops",
    "serialize_cuda",
    "spconv",
    "torch_cluster",
    "torch_scatter",
    "torch_sparse",
]

for module in modules:
    importlib.import_module(module)

assert torch.cuda.is_available()
assert torch.version.cuda == "12.4"
assert torch.cuda.get_device_capability() == (8, 0)

x = torch.randn(1024, 1024, device="cuda")
y = x @ x
torch.cuda.synchronize()

print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("device:", torch.cuda.get_device_name())
print("matrix checksum:", y.sum().item())
print("imports validated")
PY
```

Also check that the environment is synchronized:

```bash
uv sync --all-extras --locked --check
```

### 5. Run focused CUDA tests

Add the dev group without changing the training extras:

```bash
uv sync --all-extras --group dev --locked
```

Run:

```bash
uv run --no-sync pytest -q tests/test_serialize_cuda.py
```

Inspect that test file before expanding the test set.
Run additional focused extension tests only when their data and runtime requirements are understood.

### 6. Verify repeated installation

Run the full installer again with the same environment and variables:

```bash
time ./install.sh 2>&1 | tee /tmp/pimm-uv-second-install.log
```

The second run should not compile any local extension.
It should resolve and check the environment quickly.

Inspect the log:

```bash
rg -n "Building (cnms|pointgroup-ops|pointops|pointrope|pytorch3d-ops|serialize-cuda)" \
  /tmp/pimm-uv-second-install.log
```

No matches are expected.

If a package rebuilds, compare its cache keys and the values of:

```bash
env | rg '^(TORCH_CUDA_ARCH_LIST|CUDA_HOME|CC|CXX|MAX_JOBS|UV_)='
```

Changing architecture, compiler, CUDA path, or source content is expected to invalidate the corresponding wheel.

### 7. Verify the no-FlashAttention profile

Using the same full environment:

```bash
time ./install.sh --no-flash 2>&1 | tee /tmp/pimm-uv-no-flash.log
```

No local CUDA extension should rebuild.

Verify:

```bash
uv run --no-sync python - <<'PY'
import importlib
import importlib.util

assert importlib.util.find_spec("flash_attn") is None

for module in [
    "cnms",
    "pointgroup_ops",
    "pointops",
    "pointrope",
    "pytorch3d_ops",
    "serialize_cuda",
]:
    importlib.import_module(module)

print("no-FlashAttention profile validated")
PY
```

Restore the full profile:

```bash
./install.sh
```

FlashAttention should be restored from the wheel without source compilation.

### 8. Verify a separate launcher-only environment

Keep this separate from the training environment:

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-cluster-launcher"
./install.sh --launcher-only
```

Verify:

```bash
.venv-cluster-launcher/bin/pimm --help
.venv-cluster-launcher/bin/python - <<'PY'
import importlib.util

assert importlib.util.find_spec("torch") is None
print("launcher-only environment validated")
PY
```

Verify a launch plan without starting training:

```bash
UV_PROJECT_ENVIRONMENT="$PWD/.venv-cluster-launcher" \
  uv run --no-sync pimm launch --dry-run \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

Restore the full environment variable before subsequent tests:

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-cluster-full"
```

### 9. Build the final standard Docker image

If Docker and BuildKit are available on the cluster:

```bash
docker buildx build \
  --load \
  --platform linux/amd64 \
  --build-arg TORCH_CUDA_ARCH_LIST=8.0 \
  -f .github/docker/Dockerfile \
  -t pimm:uv-cluster-test \
  .
```

The final Dockerfile must complete one locked sync.
It must not perform a second native extension build.

Validate the image:

```bash
docker run --rm --gpus all pimm:uv-cluster-test bash -lc '
  uv sync --project /opt/pimm/src --all-extras --group docs --locked --check
  python - <<'"'"'PY'"'"'
import importlib
import torch

for module in [
    "cnms",
    "flash_attn",
    "pointgroup_ops",
    "pointops",
    "pointrope",
    "pytorch3d_ops",
    "serialize_cuda",
    "spconv",
    "torch_cluster",
    "torch_scatter",
    "torch_sparse",
]:
    importlib.import_module(module)

assert torch.cuda.is_available()
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
PY
'
```

Rebuild the unchanged image:

```bash
docker buildx build \
  --load \
  --platform linux/amd64 \
  --build-arg TORCH_CUDA_ARCH_LIST=8.0 \
  -f .github/docker/Dockerfile \
  -t pimm:uv-cluster-test \
  .
```

Every expensive step should report `CACHED`.

### 10. Build docs with the final image

Run the same shape as docs CI:

```bash
docker run --rm --gpus all \
  -v "$PWD":/work \
  -w /work \
  -e PYTHONPATH=/work \
  pimm:uv-cluster-test \
  bash -lc "uv run --project /opt/pimm/src --no-sync make -C docs html"
```

Confirm that `docs/build/html/index.html` exists and that registry generation imports the mounted checkout.

### 11. Build and validate the NERSC image

If Docker is available:

```bash
docker buildx build \
  --load \
  --platform linux/amd64 \
  --build-arg TORCH_CUDA_ARCH_LIST=8.0 \
  -f .github/docker/Dockerfile.nersc \
  -t pimm:uv-nersc-test \
  .
```

Validate MPI-enabled h5py:

```bash
docker run --rm pimm:uv-nersc-test python - <<'PY'
import h5py
import mpi4py

assert h5py.get_config().mpi
print("h5py:", h5py.__version__)
print("mpi4py:", mpi4py.__version__)
print("parallel h5py validated")
PY
```

Validate two MPI ranks if the container runtime permits it:

```bash
docker run --rm pimm:uv-nersc-test \
  mpiexec -n 2 python -c \
  'from mpi4py import MPI; print(MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size)'
```

If the cluster uses Apptainer or Shifter rather than Docker, adapt these runtime checks to the site's supported image path.

### 12. Check workflows

Run actionlint if available:

```bash
actionlint .github/workflows/docker.yml .github/workflows/docs.yml
```

The local machine did not have actionlint installed.
Dockerfile structural checks passed locally except for the expected warning that an amd64 base image was being checked from an arm64 host.

### 13. Final repository checks

Run:

```bash
uv lock --check
git diff --check
bash -n install.sh .github/docker/common/install_system.sh .github/docker/entrypoint.sh
git status --short
```

Use the IDE linter on every Python file changed during cluster fixes.

Do not add generated environments, caches, docs output, `.DS_Store`, or test logs to version control.

## Known risks and review points

### PyTorch base image duplication

The Dockerfiles still use `pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel`.
uv installs its own locked PyTorch into `/opt/pimm/.venv`, so the image contains both the base image's Conda environment and the uv environment.

This is correct but increases image size.
After functionality is fully validated, consider changing the base to an NVIDIA CUDA 12.4 cuDNN development image.
Do not make that change during initial cluster validation because it adds another independent variable.

### pimm source changes and Docker cache scope

The current single-sync Docker design copies `pimm/` before the uv sync.
A pimm source change therefore invalidates the sync layer.

The BuildKit uv cache may reduce downloads, but local path dependencies can still be rebuilt in a fresh environment.
Measure this behavior on native Linux before redesigning it.

### Docs image publication order

The docs workflow currently assumes the published project image already contains uv and the docs dependency group.
The first merge of this migration can race the image publication workflow.
Resolve this before treating CI migration as complete.

### BuildKit `COPY --exclude`

The Dockerfiles use `COPY --exclude`.
Docker Desktop accepted this syntax through `# syntax=docker/dockerfile:1`.
Confirm that the reusable GitHub Docker builder uses a recent Dockerfile frontend and BuildKit version.

### NERSC source builds

The NERSC image rebuilds the locked mpi4py and h5py versions from source.
That image path has not yet completed end-to-end validation.
Pay attention to MPICH ABI compatibility, HDF5 discovery, and uv's `--no-binary` option behavior.

## Useful failure signatures

If uv reports that a local package cannot use `match-runtime`, verify that the package has static `[project]` name, version, and dependencies in its own `pyproject.toml`.

If a PyTorch extension cannot import `torch` during its build, verify its root `[tool.uv.extra-build-dependencies]` entry.

If PyTorch3D reports an absolute setup source path, verify that its `sources` values remain relative.

If PyTorch3D cannot find `utils/pytorch3d_cutils.h`, verify that `include_dirs` remains absolute.

If `cc1plus` is killed, reduce `MAX_JOBS` to 1 and verify `concurrent-builds = 1`.

If the installer rejects CUDA, compare the exact `nvcc --version` release with 12.4.

If the installer rejects the compiler, select GCC and G++ major version 9 through 12 through modules or `CC` and `CXX`.

If sparsehash validation fails, provide the directory containing `google/sparse_hash_map` through the site's module system or `CPLUS_INCLUDE_PATH`.

If a repeated install rebuilds extensions, compare all declared cache keys and verify that the environment path and uv cache were preserved.

If docs import the image's source instead of the mounted checkout, verify `PYTHONPATH=/work`.

## Continuation prompt

The following prompt can be pasted into a new Cursor session:

```text
Continue the uv installation migration in this repository.
Read UV_MIGRATION_CLUSTER_HANDOFF.md first and follow its cluster validation runbook.
Do not edit the existing plan file.
Preserve all current tracked and untracked changes because the branch also contains an unrelated documentation rewrite.
Start by confirming the branch, base revision, git status, CUDA 12.4 toolchain, compiler version, sparsehash headers, A100 visibility, uv lock, and shell syntax.
Then complete the remaining validation todo without stopping: clean full install, repeated install without native rebuilds, no-FlashAttention profile, launcher-only profile, focused CUDA tests, final standard image, docs build, and NERSC MPI image.
Fix failures within the migration scope, update this handoff when the state changes, and do not commit unless explicitly requested.
```
