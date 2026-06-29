# Installation

Pick one of three ways to get the environment. **All three share one source of
truth** for dependencies under `.github/docker/`:

- `common/versions.sh` — the pins (torch, CUDA, flash-attn, arch list)
- `requirements/{requirements,requirements-cuda,requirements-dev}.txt`
- `common/install_*.sh` — the install steps

The Docker image build and the local conda installer (`install.sh`) run the
**same** scripts, so the container and the conda env resolve to the same
torch / CUDA / extension stack.

Pre-built images are on Docker Hub:

| Image | Description |
|-------|-------------|
| `youngsm/pimm:pytorch2.5.0-cuda12.4` | Standard image (`:main` tracks latest `main`) |
| `youngsm/pimm-nersc:pytorch2.5.0-cuda12.4` | NERSC/Shifter variant: MPI-aware HDF5 + `mpi4py` |

```bash
git clone https://github.com/DeepLearnPhysics/particle-imaging-models.git
cd particle-imaging-models
```

## Option A — Singularity / Apptainer (recommended on HPCs)

First, pull the container from Dockerhub and rebuilt as a singularity image (`pimm.sif`):

```bash
apptainer pull pimm.sif docker://youngsm/pimm:pytorch2.5.0-cuda12.4
```

Run pimm from your clone directory — the container uses your checkout as the pimm
source. Run a command directly:

```bash
apptainer run --nv pimm.sif \
  pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

…or open a shell and work inside:

```bash
apptainer run --nv pimm.sif
# now inside the container:
pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

For batch jobs you don't enter the container yourself — `pimm submit` wraps
`train.sh` in the container for you. See {doc}`../hpc/sites`.

## Option B — Docker (local dev, or building images)

Build the image, then run pimm from your clone directory — the container uses
your checkout as the pimm source. Run a command directly:

```bash
# build (standard; NERSC variant: -f .github/docker/Dockerfile.nersc)
docker build -f .github/docker/Dockerfile -t pimm:local .

docker run --rm --gpus all -v "$PWD:$PWD" -w "$PWD" pimm:local \
  pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

…or open a shell and work inside:

```bash
docker run --rm -it --gpus all -v "$PWD:$PWD" -w "$PWD" pimm:local bash
# now inside the container:
pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

Build args default to those found in `.github/docker/common/versions.sh`: `TORCH_VERSION`,
`CUDA_VERSION`, `CUDA_VERSION_NO_DOT`, `CUDNN_VERSION`, `TORCH_CUDA_ARCH_LIST`.
The entrypoint sets `PYTHONNOUSERSITE=1` and neutralizes host conda/mamba shell
hooks, so bind-mounted home directories don't shadow the image's packages.

## Option C — Local conda env (no container)

`install.sh` creates a conda env using the same `.github/docker` scripts the
image build uses. conda provides only the toolchain (nvcc + matching compilers +
sparsehash); pip installs torch and everything else.

```bash
./install.sh                 # GPU env, builds flash-attn (matches the image)
#   ./install.sh --no-flash  # skip the slow flash-attn source build
#   ./install.sh --cpu       # CPU-only: skip CUDA wheels/extensions
#   ./install.sh --name foo  # custom env name (default: pimm-torch2.5.0-cu124)
conda activate pimm-torch2.5.0-cu124
```

It installs torch + torchvision (pinned), the python deps, the PyG/spconv CUDA
wheels, the local CUDA extensions under `libs/`, and `pimm` editable.

:::{note}
**A GPU is not required to install** — the CUDA extensions cross-compile for
`TORCH_CUDA_ARCH_LIST` (defaulted in `versions.sh`) via `nvcc`; no device is
touched at build time. Set it for your hardware if it's not in the default list,
e.g. `TORCH_CUDA_ARCH_LIST="8.6" ./install.sh`. A GPU is only needed to *train*.
FlashAttention is optional — `--no-flash` skips it, and configs run with
`enable_flash=False`.
:::

:::{note}
**CUDA point-serialization backend.** One of the `libs/` extensions,
`serialize_cuda` (vendored from
[`point_serialization_cuda`](https://github.com/ChristianSchott/point_serialization_cuda)),
provides CUDA kernels for z-order / Hilbert serialization. When built, it is a
transparent drop-in: `pimm.models.utils.serialization` routes CUDA tensors
through it (~20% faster PTv3 inference, bit-for-bit identical to the PyTorch
encoders) and falls back to PyTorch otherwise. Set
`PIMM_DISABLE_SERIALIZE_CUDA=1` to force the PyTorch path.
:::

### Editable install everywhere you launch

Because `pimm` is a console-script entry point, every host where you run
`pimm launch` / `pimm submit` needs the checkout installed in the active
environment (the options above all do this):

```bash
pip install -e .
# equivalent without an editable install:
python -m pimm.cli launch ...
```

Confirm:

```bash
python -c "import pimm; print(pimm.__file__)"
pimm launch --help
```

## Verify the install

A fast, light first run that builds the model, dataloader, and trainer and runs a
couple of steps. The `max_len` flags **cap** how many events are used to keep the
run short — they do *not* remove the data requirement. This command needs
**PILArNet-M (v1) on disk and an NVIDIA GPU**:

```bash
pimm launch \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask \
  -- epoch=1 data.train.max_len=32 data.val.max_len=16 \
     batch_size=4 num_worker=0 use_wandb=False
```

If this runs an epoch and writes an `exp/.../` directory, your install is good.
`pimm launch` runs on the current node and auto-detects all visible GPUs by
default.

:::{note}
**Prerequisites.** No data yet? Fetch a revision first with `python
scripts/download_pilarnet.py --version v1` (see {doc}`../datasets/pilarnet`). No
GPU on this machine? The run starts but `torchrun` exits with `RuntimeError: no
CUDA devices available` — run it on a GPU node (see {doc}`../hpc/index`). There is
currently no data-free or CPU-only smoke test.
:::

:::{note}
**Prerequisites.** No data yet? Fetch a revision first with `python
scripts/download_pilarnet.py --version v1` (see {doc}`../datasets/pilarnet`). No
GPU on this machine? The run starts but `torchrun` exits with `RuntimeError: no
CUDA devices available` — run it on a GPU node (see {doc}`../hpc/index`). There is
currently no data-free or CPU-only smoke test.
:::

## Environment variables

Scripts load a `.env` from the repo root if present (existing shell variables
win):

```bash
cp example.env .env
```

| Variable | Purpose |
|----------|---------|
| `PILARNET_DATA_ROOT_V1`, `_V2` | PILArNet-M data roots per revision |
| `MODEL_DIR` | Redirect (large) checkpoints to another filesystem; the experiment `model/` becomes a symlink |
| `WANDB_API_KEY` | Alternative to `wandb login` |
| `TORCH_CUDA_ARCH_LIST` | Target GPU archs for building the CUDA extensions |

`PILArNetH5Dataset` also falls back to `~/.cache/pimm/pilarnet/<revision>` when
no data-root variable is set. See {doc}`../datasets/pilarnet` for downloading.

## Troubleshooting

:::{dropdown} `pimm: command not found`
Run `pip install -e .` in the active environment, or use `python -m pimm.cli
launch ...`. If imports resolve to the wrong checkout, inspect `PYTHONPATH` —
non-dev training runs intentionally use the copied snapshot under
`exp/.../code`; pass `--train.no-code-copy` (or `train.sh -C`) to run from the
live source.
:::

:::{dropdown} Sparse extension imports fail
Verify these match each other: the PyTorch + CUDA versions, `spconv-cu124` (or
the wheel matching your CUDA stack), the built output under `libs/pointops`,
`libs/pointgroup_ops`, `libs/cnms`, `libs/pytorch3d_ops`, and your
`TORCH_CUDA_ARCH_LIST` for the target GPUs. Make sure `ninja` is installed and
CUDA 12.4 tooling is active.
:::

:::{dropdown} A model import fails on optional CUDA dependencies
Importing `pimm.models` pulls in many model families, some with heavy CUDA
deps. Tests sometimes import a narrower module to avoid this. For inference you
usually only need the one family you are loading.
:::

## Next steps

- {doc}`quickstart` — run a real job and learn the launcher flags.
- {doc}`concepts` — the packed-tensor / registry / config mental model.
