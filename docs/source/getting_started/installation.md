# Installation

pimm has two layers with different dependency needs:

- The **CLI / launcher**
- The **full training stack** (PyTorch, CUDA, spconv, FlashAttention, the local CUDA
  extensions) is heavy and is what actually runs on the GPU nodes. 

:::{important}
**Before you start — what the training stack needs:**

- **Linux + an NVIDIA GPU (CUDA >12.4).** There is no CPU / macOS / Apple-Silicon
  training path — the sparse kernels (spconv, FlashAttention, the `libs/` CUDA
  extensions) require CUDA.
- **For the from-source build:** the CUDA toolkit (`nvcc`) and `ninja`; the local
  extension build is slow and GPU-arch-specific (`TORCH_CUDA_ARCH_LIST`).

The pure-Python launcher (`pimm launch` / `submit` / `export`) is light and runs
anywhere. Only the GPU nodes need the full stack.
:::

## Option A — Container (recommended)

Pre-built images live on Docker Hub:

| Image | Description |
|-------|-------------|
| `youngsm/pimm:main` | Standard image |
| `youngsm/pimm-nersc:main` | NERSC/Shifter variant with MPI-aware HDF5 + `mpi4py` |

```bash
git clone https://github.com/DeepLearnPhysics/particle-imaging-models.git
cd particle-imaging-models
apptainer pull /path/to/pimm.sif docker://youngsm/pimm:main
```

The image installs `pimm` as an editable package at `/opt/pimm/src`. **Bind your
own clone over that path** so the `pimm` command and imports resolve to your
local code, not the build baked into the image:

```bash
apptainer exec --nv \
  --bind "$PWD:/opt/pimm/src" \
  --pwd /opt/pimm/src \
  /path/to/pimm.sif \
  pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

The managed launch path (`pimm submit`) wires this mount automatically — it
binds `paths.repo_root` onto `container.repo_mount` (default `/opt/pimm/src`).
See {doc}`../hpc/sites`.

:::{dropdown} Building the image yourself
The generic image is built from `docker/Dockerfile`; the NERSC variant from
`docker/Dockerfile.nersc` (it builds MPICH + parallel HDF5 from source so
Shifter can swap in Cray MPICH at runtime).

```bash
docker build \
  --build-arg TORCH_VERSION=2.5.0 \
  --build-arg CUDA_VERSION=12.4 \
  --build-arg CUDA_VERSION_NO_DOT=124 \
  --build-arg 'TORCH_CUDA_ARCH_LIST=8.0 8.6 8.9 9.0' \
  -f docker/Dockerfile -t pimm:dev .
```

The container entrypoint sets `PYTHONNOUSERSITE=1` and shadows host conda/mamba
shell hooks, which keeps imports correct when home directories are bind-mounted
by Apptainer/Singularity.
:::

## Option B — From source (conda)

`environment.yml` is the authoritative recipe for the local GPU stack: Python
3.10, PyTorch 2.5.0 / CUDA 12.4, GCC/GXX 13.2 for extension builds, and sparse
point-cloud dependencies (PyG, spconv, ocnn, FlashAttention, CLIP) plus the
local CUDA extensions under `libs/`.

```bash
git clone https://github.com/DeepLearnPhysics/particle-imaging-models.git
cd particle-imaging-models

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0 8.6 8.9 9.0}"
conda env create -f environment.yml
conda activate pimm-torch2.5.0-cu12.4
pip install -e .
```

Create the environment **from the repository root** so the relative paths in the
pip section resolve — it installs the local extensions `libs/pointrope`,
`libs/pointops`, `libs/pointgroup_ops`, `libs/cnms`, and `libs/pytorch3d_ops`.

:::{note}
The `TORCH_CUDA_ARCH_LIST` export lets the LitePT / PointROPE CUDA extension
build on a login or container-build host that has **no visible GPU**.
FlashAttention needs CUDA 11.6+. If it is unavailable, set `enable_flash=False`
in your config's backbone block.
:::

### Editable install everywhere you launch

Because `pimm` is a console-script entry point, every host where you run
`pimm launch` / `pimm submit` needs the checkout installed in the active
environment:

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

`PILArNetH5Dataset` also falls back to `~/.cache/pimm/pilarnet/<revision>` when
no data-root variable is set. See {doc}`../datasets/pilarnet` for downloading, and
{doc}`../reference/environment` for the full variable list.

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
`TORCH_CUDA_ARCH_LIST` / site `CUMM_CUDA_ARCH_LIST` for the target GPUs. Make
sure `ninja` is installed and CUDA 12.4 tooling is active.
:::

:::{dropdown} A model import fails on optional CUDA dependencies
Importing `pimm.models` pulls in many model families, some with heavy CUDA
deps. Tests sometimes import a narrower module to avoid this. For inference you
usually only need the one family you are loading.
:::

## Next steps

- {doc}`quickstart` — run a real job and learn the launcher flags.
- {doc}`concepts` — the packed-tensor / registry / config mental model.
