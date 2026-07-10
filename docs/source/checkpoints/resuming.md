# Resuming a run

Resuming applies to **every** run, not just HPC jobs - a laptop run killed with
Ctrl-C resumes the same way a preempted Slurm job does. pimm distinguishes three
things that are easy to conflate:

```{list-table}
:header-rows: 1
:widths: 24 38 38

* - Action
  - What it restores
  - Use when
* - **Resume**
  - *Everything* - model, optimizer, scheduler, AMP scaler, RNG, dataloader
    position, step, samples-seen, best-metric
  - Continuing the *same* run (e.g. after a crash or timeout)
* - **Fine-tune**
  - *Model weights only* (optionally remapped)
  - Starting a *new* run/task from a pretrained checkpoint
* - **Reshard**
  - Everything, re-distributed to a new world size
  - Resuming on a different number of GPUs
```

Fine-tuning (weights-only warm-start) and key remapping are covered in
{doc}`saving_and_loading`. This page is about resuming and resharding the *same*
run.

## Resume the same run

Set `resume=True` and point at the existing experiment. Locally:

```bash
pimm launch --train.resume \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask \
  --run.name my-run
```

On Slurm it's the same flag through `pimm submit` (see {doc}`../hpc/resuming`).

Resume is **exact**. When `resume=True`,
{py:class}`~pimm.engines.hooks.checkpoint.CheckpointLoader` restores model,
optimizer, scheduler, AMP scaler, trainer progress (`start_epoch`, `start_iter`,
`global_step`, `samples_seen`, `best_metric_value`), the stateful dataloader
position, and RNG (Python / NumPy / CPU / all CUDA) - per rank. Combined with
`seed` and `deterministic=True`, the continued trajectory matches what an
uninterrupted run would have produced.

Two important details:

- **It reuses the saved config.** Resume loads `exp/<group>/<name>/config.py`
  (the resolved artifact), *not* the original file under `configs/`. Edits to the
  original config after the run started do **not** take effect on resume.
- **It reuses the code snapshot.** The run continues from
  `exp/<group>/<name>/code/`, the snapshot taken at first launch - so behavior is
  reproducible even if your working tree changed.

The experiment directory must already exist; the launcher picks the newest
*complete* checkpoint among `model/last`, `model/last.prev`,
`model/model_last.pth`. If none is complete, it exits rather than silently
restarting.

If a checkpoint was saved mid-epoch and contains dataloader state, the loader
restores that position and enumerates the loader with `start=start_iter` - you
continue from the exact batch. If the checkpoint is mid-epoch but lacks
dataloader state (e.g. a legacy single-file checkpoint), the loader **warns and
replays from the start of the saved epoch**.

:::{note}
A checkpoint taken after a step resumes after that step - it neither repeats
nor skips the batch.
:::

## Resume on a different number of GPUs

You started an 8-GPU run; now you only have 4 (or want to scale up). What happens
depends on the checkpoint format ({doc}`index`).

### The default format reshards automatically

With the `standard`/DCP format (the default for multi-rank / requeued / FSDP2
runs), just change the resource flag and resume:

```bash
# Started on 8 GPUs; resume on 4 - no extra flags.
pimm submit --site mycluster \
  --resources.nnodes 1 --resources.nproc-per-node 4 \
  --train.config <cfg> --run.name my-run --train.resume
```

This works because:

- The trainer state lives in a **Distributed Checkpoint** (`trainer.dcp/`), which
  resharded reads/writes natively - model, optimizer, scheduler, step,
  samples-seen, and best-metric all redistribute cleanly.
- The **global batch size is fixed**, so iterations-per-epoch is identical
  regardless of GPU count. Your schedule and accounting stay aligned.

### Legacy format

The legacy single-file format does **not** reshard. Strict resume
(`resume_strict_state=True`, the default) refuses to remap per-rank dataloader /
RNG state saved under a different world size, and raises. To resume anyway:

```bash
pimm launch --train.config <cfg> --run.name my-run --train.resume \
  -- resume_strict_state=False
```

This is safe specifically because the global batch size is fixed, so
iters/epoch is identical - only the per-rank RNG/dataloader bookkeeping is being
relaxed.

:::{warning}
`resume_strict_state=False` trades *exact* data/RNG resume for the ability to
change GPU count with the legacy format. The continued trajectory is no longer
bitwise-identical to an uninterrupted run. Prefer the `standard`/DCP format when
you anticipate changing world size.
:::

### Which format am I using?

```text
model/last/trainer.dcp/   present  →  standard / DCP  (reshards automatically)
model/model_last.pth      present  →  legacy          (needs resume_strict_state=False)
```

Force the reshardable format for new runs (the launcher already does this for
multi-rank, requeued, and FSDP2 runs):

```bash
pimm launch --train.config <cfg> -- checkpoint_format=standard
```

## Troubleshooting

:::{dropdown} "It restarted from scratch instead of resuming"
Check that a *complete* checkpoint exists
(`python -m pimm.utils.path latest-checkpoint exp/.../run`). A `standard`/DCP
checkpoint needs its `.complete` marker; an interrupted save leaves only the
`.tmp` dir and the previous good `.prev`.
:::

:::{dropdown} "My config change didn't take effect on resume"
Resume uses the saved `exp/.../config.py`. To change behavior, either start a
new run, or edit the saved artifact directly (advanced) - but prefer a new run
with a child config for anything you want to track.
:::

:::{dropdown} "A custom dataset/hook isn't found on resume"
Resume rebuilds everything from the dumped config, so the class has to be
registered the same way every time pimm starts. Register custom datasets/hooks in
the package `__init__.py` so their decorator side effects always run.
:::

:::{dropdown} "World-size mismatch error on resume"
Strict resume (`resume_strict_state=True`) refuses to remap per-rank
dataloader/RNG state saved under a different world size. Use the DCP/standard
format to reshard, or set `resume_strict_state=False` for the legacy format.
:::
