# Environment variables

The consolidated list of environment variables pimm reads. Scripts
(`scripts/train.sh` / `test.sh`) source a repo-root `.env` if present (existing
shell variables win), and {doc}`site profiles <../hpc/sites>` inject their own
`env:` block into batch jobs.

## Data

| Variable | Purpose |
|----------|---------|
| `PILARNET_DATA_ROOT_V1` / `_V2` / `_V3` | PILArNet-M data root per revision. `PILArNetH5Dataset` falls back to `~/.cache/pimm/pilarnet/<revision>` when unset. See {doc}`../datasets/pilarnet`. |

## Checkpoints & logging

| Variable | Purpose |
|----------|---------|
| `MODEL_DIR` | Redirect (large) checkpoints to another filesystem; the experiment `model/` becomes a symlink to `${MODEL_DIR}/exp/<group>/<name>/model/`. |
| `WANDB_API_KEY` | Weights & Biases auth (alternative to `wandb login`). |

## Hugging Face downloads

Used when loading `hf://` weights (see {doc}`../checkpoints/huggingface`):

| Variable | Purpose |
|----------|---------|
| `PIMM_HF_CACHE` | Where pimm caches `hf://` downloads (checked first). |
| `HF_HOME` / `HF_HUB_CACHE` | Standard Hugging Face cache locations, used when `PIMM_HF_CACHE` is unset. |

## Distributed (set by the launcher)

`torchrun` / Slurm set these — you normally don't set them yourself, but they're
useful when debugging multi-node runs (see {doc}`../distributed/index`):

`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `LOCAL_WORLD_SIZE`, `GROUP_RANK`,
`MASTER_ADDR`, `MASTER_PORT` — derived from the `SLURM_*` job variables on a
cluster.

## Build (from-source / container)

| Variable | Purpose |
|----------|---------|
| `TORCH_CUDA_ARCH_LIST` / `CUMM_CUDA_ARCH_LIST` | Target GPU compute capabilities when building the CUDA extensions (see {doc}`../getting_started/installation`). |
| `OMP_NUM_THREADS` | Set by the job to `cpus_per_proc` (see {doc}`../hpc/sites`). |
| `PYTHONNOUSERSITE` | Set by the container entrypoint to keep imports correct under bind-mounted homes. |

## See also

- {doc}`../getting_started/installation` — where the data / W&B vars are first used.
- {doc}`../hpc/sites` — site `env:` blocks and `.env` for cluster-wide values.
- {doc}`../datasets/pilarnet` — data roots and the cache fallback.
