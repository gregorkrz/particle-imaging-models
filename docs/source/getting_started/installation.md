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
- An NVIDIA GPU and a compatible driver.
- The CUDA 12.4 toolkit, including `nvcc`.
- GCC and G++ 9 through 12.
- The sparsehash headers.

uv installs Python 3.10, CMake, Ninja, PyTorch, and the remaining Python packages.
uv does not install the CUDA toolkit, host compiler, or sparsehash because they are system-level build dependencies.

On Ubuntu, the compiler and sparsehash headers are available through:

```bash
sudo apt update
sudo apt install build-essential libsparsehash-dev
```

Install CUDA 12.4 using the [NVIDIA CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/), or load the CUDA and compiler modules provided by your cluster.
For example:

```bash
module load cuda/12.4 gcc/12
```

The installer checks these requirements before downloading the training environment.
It does not invoke `sudo` or modify system packages.

## Local uv environment

Run the installer from the repository root:

```bash
./install.sh
```

The installer:

1. Installs uv with the official Astral installer if uv is not already available.
2. Validates CUDA 12.4, the host compiler, sparsehash, and the visible NVIDIA GPUs.
3. Detects the compute capability of each visible GPU.
4. Creates `.venv` and installs the versions recorded in `uv.lock`.
5. Builds the six local CUDA extensions for the detected GPU architectures.
6. Verifies the CUDA packages and extensions can be imported.

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

### Repeated installs and extension rebuilds

uv reuses downloaded packages and compiled extension wheels.
Running the installer again does not rebuild unchanged extensions.

Each local extension tracks its C++, CUDA, header, Python, compiler, CUDA path, and `TORCH_CUDA_ARCH_LIST` inputs.
Changing one of those inputs rebuilds only the affected package.
To force one extension to rebuild, run:

```bash
uv sync --all-extras --locked --reinstall-package pointops
```

Set `TORCH_CUDA_ARCH_LIST` before installation when building for a GPU that is not visible on the build host:

```bash
TORCH_CUDA_ARCH_LIST="8.0 9.0" ./install.sh
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
- `TORCH_CUDA_ARCH_LIST` overrides automatic GPU architecture detection.
- `CUDA_HOME` selects the CUDA 12.4 toolkit used for extension builds.
- `CC` and `CXX` select the host compilers.
- `MAX_JOBS` controls native build parallelism and defaults to 2 in `install.sh`.
- `PIMM_DISABLE_SERIALIZE_CUDA=1` disables the optional CUDA serialization backend.

## Troubleshooting

:::{dropdown} `nvcc` is missing or reports another version
Install CUDA 12.4 or load the matching cluster module.
If more than one toolkit is installed, set `CUDA_HOME` before running `install.sh`.
:::

:::{dropdown} sparsehash is missing
Install `libsparsehash-dev` on Debian or Ubuntu.
On another distribution, install the package that provides `google/sparse_hash_map`.
:::

:::{dropdown} an extension was built for the wrong GPU
Set `TORCH_CUDA_ARCH_LIST` explicitly and reinstall that package with `uv sync --all-extras --locked --reinstall-package <package>`.
:::

:::{dropdown} a bare `uv sync` removed training packages
Optional training dependencies are selected by `--all-extras`.
Restore the full environment with `uv sync --all-extras --locked`.
:::

## Next steps

- {doc}`quickstart` covers a complete training and fine-tuning run.
- {doc}`concepts` explains packed tensors, registries, and Python configs.
