# Chaining, walltime & QOS

Most HPC queues cap walltime well below the time a foundation-model run needs.
pimm's answer is **requeue chaining**: submit one job that automatically
requeues and resumes from the latest checkpoint when it times out.

## How chaining works

`--chain.jobs N` submits a *single* submitit job with
`slurm_max_num_timeout = N - 1`. On timeout, submitit requeues the next
**pre-rendered attempt** in the same allocation.

```text
--chain.jobs 4, --resources.time 02:00:00
┌──────────┐ timeout ┌──────────┐ timeout ┌──────────┐ timeout ┌──────────┐
│ attempt1 │ ──────▶ │ attempt2 │ ──────▶ │ attempt3 │ ──────▶ │ attempt4 │
│ (fresh)  │ resume  │ (resume) │ resume  │ (resume) │ resume  │ (resume) │
└──────────┘         └──────────┘         └──────────┘         └──────────┘
   2h max               2h max               2h max               2h max
```

```bash
pimm submit --site s3df --recipe launch/runs/e050_tail.yaml \
  --chain.jobs 4 \
  --run.name e050-tail-chain \
  --resources.time 02:00:00
```

Behavior (`build_attempts` in `pimm/launch/submit.py`):

- All attempts share the **same experiment name and directory**.
- Attempt 1 starts normally unless resume is requested; attempts 2+ resume
  (`-r true`) from the newest complete checkpoint.
- W&B runs are named `<base>-job0001`, `<base>-job0002`, … and grouping fields
  (`wandb_group`, `wandb_job_type`, `wandb_job_index`, `chain_jobs`) are threaded
  through as training overrides.
- The `Chain` dataclass has only `jobs`, `resume_first`, `wandb_group`, and
  `wandb_job_type`.

:::{important}
This is **submitit requeue, not a Slurm dependency chain**. There is no
`afterany`/`afterok` chain and no `chain.dependency` setting. Chaining is
batch-only (`pimm submit`) — `--interactive` with `chain.jobs > 1` is rejected.
:::

### Resuming an existing experiment as a fresh chain

Use `chain.resume_first` when you start a *new* chain that should resume an
already-started experiment from its first attempt:

```bash
pimm submit --site s3df --recipe launch/runs/e050_tail.yaml \
  --chain.jobs 3 --run.name existing-run --chain.resume-first
```

### `slurm.dependency` is separate

`slurm.dependency` is a single pass-through value handed to submitit's
`slurm_dependency`; it is **not** chain-managed. Use it for one-off
"start after job X" ordering, not for walltime chaining.

## Walltime

Set per-attempt walltime with `--resources.time HH:MM:SS`. Pick a value the
queue schedules quickly, then use `--chain.jobs` to reach the total training
budget. A 4×2h chain trains for ~8h of wall time but only ever asks the
scheduler for 2h slots.

:::{tip}
Checkpoint cadence and walltime should be friends. With short attempts, make
sure `CheckpointSaverIteration.save_freq` is small enough that an attempt always
leaves a recent complete checkpoint before it times out. See
{doc}`../checkpoints/index`.
:::

## QOS, accounts, and partitions

These are site fields you override per invocation:

```{list-table}
:header-rows: 1
:widths: 30 30 40

* - Setting
  - Flag
  - Notes
* - QOS
  - `--slurm.qos QOS`
  - e.g. NERSC `regular`, `interactive`, `shared_interactive`
* - Account
  - `--slurm.account ACCT`
  - S3DF default `mli:nu-ml-dev`; NERSC default `m5238_g`
* - Partition
  - `--slurm.partition PART`
  - S3DF default `ampere`
* - Constraint
  - site YAML (`constraint`)
  - NERSC default `gpu`
```

```bash
pimm submit --site nersc \
  --slurm.qos regular \
  --slurm.account m5238_g \
  --resources.nnodes 4 --resources.nproc-per-node 4 --resources.time 02:00:00 \
  --recipe launch/runs/e050_tail.yaml --dry-run
```

:::{warning}
QOS is `slurm.qos`, **not** a resources field. For the NERSC interactive queue,
pass `--slurm.qos interactive` (or `shared_interactive`). Always confirm the
rendered `slurm_qos`/`slurm_account` in the `--dry-run` manifest before
submitting.
:::

## Recovery correctness

Chaining only works because resume works. The checkpoint backend is chosen for
you: `pimm launch`/`pimm submit` default {py:class}`~pimm.engines.hooks.checkpoint.CheckpointSaverIteration` to the
reshardable DCP/standard format when the run is multi-rank, requeued
(`chain.jobs > 1`), or `parallel.strategy=fsdp2`. You do **not** need to opt in.

To force the legacy single-file backend (rarely needed):

```bash
pimm submit --site s3df --train.config <cfg> \
  -- hooks.CheckpointSaverIteration.backend=torch
```

See {doc}`resuming` for the resume mechanics and {doc}`../checkpoints/index` for
the formats.

## Next

- {doc}`monitoring` — watch the chain progress.
- {doc}`resuming` — resume a submitted run; {doc}`../checkpoints/resuming` for what's restored.
