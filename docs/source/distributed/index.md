# Distributed training

pimm is built so that **the same config and the same command** run on one GPU,
many GPUs on one node, or many nodes - only the resource flags change.

- **Single & multi-GPU** - `torchrun` under the hood - no Slurm needed for one node.
- **Multi-node** - one Slurm task per node; `torchrun` fans out to the GPUs.
- **DDP / FSDP2** - pick a strategy in the `parallel` config block.

## The model

pimm uses `torchrun` exclusively - it is hardcoded in `scripts/train.sh`, and
there is no `distributed.launcher` config key. The rule everywhere is **one
process per GPU**:

```text
        ┌─ node 0 ─────────────┐     ┌─ node 1 ─────────────┐
torchrun│ rank0 rank1 rank2 r3 │ ... │ rank4 rank5 rank6 r7 │
        │  gpu0  gpu1  gpu2 g3 │     │  gpu0  gpu1  gpu2 g3 │
        └──────────────────────┘     └──────────────────────┘
        nproc-per-node = 4           nnodes = 2  ⇒  world_size = 8
```

pimm auto-detects either `torchrun` variables
(`RANK`, `WORLD_SIZE`, `LOCAL_RANK`, ...) or Slurm variables (`SLURM_PROCID`,
`SLURM_NTASKS`, ...), picks the CUDA device from the local rank, and initializes
an NCCL process group.
If no distributed environment is present, it runs single-process.

## Local and multi-GPU

A single GPU:

```bash
pimm launch \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask \
  --resources.nproc-per-node 1
```

Four GPUs on the current node - **no Slurm required**:

```bash
pimm launch \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask \
  --resources.nproc-per-node 4
```

Local rendezvous defaults (from `pimm/launch/local.py`) are `MASTER_ADDR=127.0.0.1`
and `MASTER_PORT=29500`. Use `--dry-run` to see the exact `torchrun` line.

## Multi-node

Two ways: run `pimm launch` inside your own allocation, or use the managed
submitit path with `pimm submit` (see {doc}`../hpc/index`).

:::{note}
`slurm` is the generic base profile; define a site profile for your own cluster - see {doc}`../hpc/sites`.
:::

::::{tab-set}

:::{tab-item} Managed (recommended)
```bash
pimm submit --site slurm \
  --resources.nnodes 2 \
  --resources.nproc-per-node 4 \
  --resources.time 02:00:00 \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

:::

:::{tab-item} Inside your own allocation
```bash
srun pimm launch \
  --resources.nnodes "$SLURM_NNODES" \
  --resources.nproc-per-node "$SLURM_GPUS_ON_NODE" \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```
:::

:::{tab-item} Hand-written sbatch
```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1      # one task per NODE, not per GPU
#SBATCH --gres=gpu:4

sh scripts/train.sh -m 2 -g 4 \
  -c panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```
:::

::::

:::{warning}
**One Slurm task per node.** `torchrun` starts one process per GPU itself, so do
not also wrap the launcher in `srun` with one task per GPU. The launcher renders
`--ntasks-per-node=1` deliberately. Slurm clusters differ in how GPUs are requested - some use `--gres=gpu:<N>`, others `--gpus-per-node=<N>`; check your cluster's documentation.
:::

## Global batch sizes split automatically

You configure **global** batch sizes and worker counts; `default_setup()`
derives the per-rank values:

```python
num_worker_per_gpu      = num_worker // world_size
batch_size_per_gpu      = batch_size // world_size
batch_size_val_per_gpu  = batch_size_val // world_size   # or 1 if unset
batch_size_test_per_gpu = batch_size_test // world_size  # or 1 if unset
```

:::{important}
The global `batch_size` (and val/test sizes, when set) **must divide the world
size**. Because the global batch is fixed, the number of iterations per epoch is
identical regardless of GPU count - which is exactly what makes resume across a
different world size safe (see below).
:::

Process RNG seeds are set to `seed + rank * num_worker_per_gpu`, with
`deterministic` applied when requested.

## Parallel strategies

The `parallel` block selects the strategy, defaulting to `ddp`, which wraps the model.
Tensor and pipeline parallelism are not pimm concepts; only DDP and FSDP2 are supported.

```{list-table}
:header-rows: 1
:widths: 16 84

* - Strategy
  - Behavior
* - `none`
  - No wrapper (also used automatically when `world_size == 1`). Model is moved
    to the device as-is.
* - `ddp`
  - `DistributedDataParallel` with `broadcast_buffers=False`,
    `find_unused_parameters` from the top-level config, and `device_ids` set on
    CUDA. Tries `static_graph=True`, falling back for older PyTorch.
* - `fsdp2`
  - Composable PyTorch FSDP2 (`torch.distributed._composable.fsdp.fully_shard`)
    over a `("fsdp",)` device mesh. If `parallel.wrap_classes` is set, matching
    submodules are sharded first, then the root. Requires CUDA + FSDP2 support.
    Untested. Probably won't work.
```

Example config blocks:

::::{tab-set}

:::{tab-item} DDP (default)
```python
# Usually nothing to set - ddp is the default.
parallel = dict(strategy="ddp")
find_unused_parameters = False   # set True only if your graph needs it
sync_bn = False                  # converts BN → SyncBatchNorm when world_size>1
```
:::

:::{tab-item} FSDP2
```python
parallel = dict(
    strategy="fsdp2",
    wrap_classes=["Block"],   # class-name match → shard these submodules first
)
```
The launcher auto-defaults checkpointing to the reshardable DCP format when
`parallel.strategy=fsdp2`. See {doc}`../checkpoints/index`.
:::

::::

:::{tip}
`sync_bn=True` converts BatchNorm to `SyncBatchNorm` when `world_size > 1` on
CUDA. Most pimm point models normalize differently, but enable it if your model
relies on batch statistics.
:::

## Autocast / mixed precision

AMP wraps the **forward only**; backward, unscale, gradient clipping, optimizer
step, scaler update, and scheduler step run outside autocast.

```python
enable_amp = True
amp_dtype  = "bfloat16"   # the supported value in the engine
```

With AMP on, the engine builds a `GradScaler` and skips the scheduler step when
scaler overflow prevents an optimizer step.

## Deterministic checkpointing & resume across world size

pimm checkpoints the **full** training state - model, optimizer, scheduler, AMP scaler, RNG (Python/NumPy/CPU/all CUDA), the stateful dataloader position, global step, and samples-seen - per rank.

Because the default `standard` format stores the trainer state as a
[Distributed Checkpoint (DCP)](https://pytorch.org/docs/stable/distributed.checkpoint.html),
it **reshards automatically** - resume an 8-GPU run on 4 GPUs (or vice versa)
by changing only the resource flag:

```bash
pimm submit --site slurm --resources.nnodes 1 --resources.nproc-per-node 4 \
  --train.config <cfg> --run.name <existing-run> --train.resume
```

See {doc}`../checkpoints/resuming` for how resharding works.

Full details - formats, atomic publish, the `.complete` marker, mid-epoch
semantics - are in {doc}`../checkpoints/index`.
