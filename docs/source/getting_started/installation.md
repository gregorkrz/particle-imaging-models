# Installation

pimm uses [uv](https://docs.astral.sh/uv/) for Python, dependency resolution, and the project environment.
The exact dependency set is stored in `uv.lock` and is shared by local installs, container images, and documentation builds.

Clone the repository before choosing an installation path:

```bash
git clone https://github.com/DeepLearnPhysics/particle-imaging-models.git
cd particle-imaging-models
```

## Prerequisites

A full local training installation requires:

- Linux on x86_64.
- An NVIDIA GPU and a recent driver.

That is all. The training stack, including PyTorch and the six native CUDA
extensions, installs as prebuilt CUDA 12.4 wheels, so no CUDA toolkit, host
compiler, or system libraries are needed. uv provides Python 3.10 and the rest.
The installer does not invoke `sudo` or modify system packages.

Rebuilding a native extension from source (an occasional maintainer task) does
need the CUDA 12.4 toolkit and GCC/G++ 9 through 12.

## Local uv environment

Bootstrap everything (uv, the clone, and the environment) with one command:

```bash
curl -sSL https://raw.githubusercontent.com/DeepLearnPhysics/particle-imaging-models/main/install.sh | bash
```

Or run the installer from an existing checkout:

```bash
./install.sh
```

The installer:

1. Installs uv with the official Astral installer if uv is not already available.
2. Clones the repository if the installer is run outside a checkout.
3. Creates `.venv` and installs the prebuilt wheels recorded in `uv.lock`.
4. Verifies the CUDA packages and extensions can be imported.

Run project commands through uv without activating the environment:

```bash
uv run pimm launch --help
uv run python scripts/download_pilarnet.py --help
```

You can also activate `.venv` and use `pimm` or `python` directly:

```bash
source .venv/bin/activate
pimm launch --help
```

### FlashAttention

The default install downloads the official FlashAttention 2.7.3 wheel built for Python 3.10, PyTorch 2.5, and CUDA 12.
It does not compile FlashAttention from source.

Install without FlashAttention when every selected model has `enable_flash=False`:

```bash
./install.sh --no-flash
```

### Repeated installs

Installs pull prebuilt wheels, so running the installer again is fast and
compiles nothing. To reinstall a single package (for example after clearing a
cache), run:

```bash
uv sync --locked --reinstall-package pointops
```

## Launcher-only environment

A login or submit host that only runs `pimm submit` does not need PyTorch or the CUDA extensions.
Install the small launcher environment with:

```bash
./install.sh --launcher-only
uv run pimm submit --help
```

The full training environment remains inside the selected container on compute nodes.

## Apptainer or Singularity

The prebuilt image is the shortest path on an HPC system:

```bash
apptainer pull pimm.sif docker://youngsm/pimm:pytorch2.5.0-cuda12.4
```

Run from the repository root so the image imports the current checkout:

```bash
apptainer run --nv pimm.sif \
  pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

For managed batch jobs, `pimm submit` wraps the training command in the configured image.
See {doc}`../hpc/sites`.

## Docker

Build the standard image from the same lockfile:

```bash
docker build -f .github/docker/Dockerfile -t pimm:local .
```

Run a command from the current checkout:

```bash
docker run --rm --gpus all -v "$PWD:$PWD" -w "$PWD" pimm:local \
  pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

The uv environment is stored at `/opt/pimm/.venv`, outside `/opt/pimm/src`.
Binding a checkout over the image source therefore does not hide the installed environment.

The NERSC image adds MPICH, parallel HDF5, `mpi4py`, and an MPI-enabled build of `h5py`:

```bash
docker build -f .github/docker/Dockerfile.nersc -t pimm-nersc:local .
```

## Verify the environment

Check the locked Python and CUDA stack:

```bash
uv run python - <<'PY'
import flash_attn
import pointops
import spconv
import torch
import torch_scatter

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name())
PY
```

Then run a short training job.
This command still requires PILArNet-M v1 on disk:

```bash
uv run pimm launch \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask \
  -- epoch=1 data.train.max_len=32 data.val.max_len=16 \
     batch_size=4 num_worker=0 use_wandb=False
```

## Environment variables

Scripts load `.env` from the repository root when it exists, while existing shell variables take precedence:

```bash
cp example.env .env
```

- `PILARNET_DATA_ROOT_V1` and `PILARNET_DATA_ROOT_V2` select local PILArNet-M revisions.
- `MODEL_DIR` redirects checkpoints to another filesystem.
- `WANDB_API_KEY` authenticates Weights & Biases.
- `PIMM_DISABLE_SERIALIZE_CUDA=1` disables the optional CUDA serialization backend.

## Troubleshooting

:::{dropdown} the training packages are missing
The training stack (`train`) and FlashAttention (`flash-attn`) are default dependency groups, so a plain `uv sync --locked` installs them.
If a host only needs the launcher, `uv sync --locked --no-default-groups` installs the minimal environment instead.
:::

## Next steps

- {doc}`quickstart` covers a complete training and fine-tuning run.
- {doc}`concepts` explains packed tensors, registries, and Python configs.
