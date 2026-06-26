# Resuming

pimm distinguishes three things that are easy to conflate:

```{list-table}
:header-rows: 1
:widths: 24 38 38

* - Action
  - What it restores
  - Use when
* - **Resume**
  - *Everything* — model, optimizer, scheduler, AMP scaler, RNG, dataloader
    position, step, samples-seen, best-metric
  - Continuing the *same* run (e.g. after a timeout)
* - **Warm-start**
  - *Model weights only* (optionally remapped)
  - Starting a *new* run/task from a pretrained checkpoint
* - **Reshard**
  - Everything, re-distributed to a new world size
  - Resuming on a different number of GPUs
```

## Resume the same run

```bash
pimm submit --site s3df \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask \
  --run.name my-run \
  --train.resume
```

Resume is **exact**. When `resume=True`, {py:class}`~pimm.engines.hooks.checkpoint.CheckpointLoader` restores model,
optimizer, scheduler, AMP scaler, trainer progress (`start_epoch`, `start_iter`,
`global_step`, `samples_seen`, `best_metric_value`), the stateful dataloader
position, and RNG (Python / NumPy / CPU / all CUDA) — per rank.

Two important details:

- **It reuses the saved config.** Resume loads `exp/<group>/<name>/config.py`
  (the resolved artifact), *not* the original file under `configs/`. Edits to the
  original config after the run started do **not** take effect on resume.
- **It reuses the code snapshot.** The run continues from
  `exp/<group>/<name>/code/`, the snapshot taken at first launch — so behavior is
  reproducible even if your working tree changed.

The experiment directory must already exist; the launcher picks the newest
*complete* checkpoint among `model/last`, `model/last.prev`, `model/model_last.pth`.
If none is complete, it exits rather than silently restarting.

### Mid-epoch resume

If a checkpoint was saved mid-epoch and contains dataloader state, the loader
restores that position and enumerates the loader with `start=start_iter` — you
continue from the exact batch. If the checkpoint is mid-epoch but lacks
dataloader state (e.g. a legacy single-file checkpoint), the loader **warns and
replays from the start of the saved epoch**.

:::{note}
`TrainState.from_trainer()` records the position *after* the just-completed
batch, so a checkpoint taken after a step resumes after that step — not before
it, and not duplicating it.
:::

## Warm-start a new task from a checkpoint

This is the "same snapshot, new config/new task" path — e.g. you pretrained with
Sonata and now want to fine-tune a segmentation head from those backbone
weights. Point `--train.weight` at a checkpoint **without** `--train.resume`:

```bash
pimm submit --site s3df \
  --train.config panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-enc-upcast-fft \
  --train.weight exp/panda/pretrain/sonata-run/model/model_best.pth
```

With `resume=False`, only **model weights** are loaded — the optimizer,
scheduler, step counter, and dataloader start fresh, which is what you want for a
new task. The weight may be a local file/dir or an `hf://` URI (see
{doc}`../models/index`).

### Remapping keys across architectures

A pretraining checkpoint's keys rarely line up with a fine-tune model
one-to-one (e.g. `student.backbone.*` → `backbone.*`). The `CheckpointLoader`
hook remaps them. Model keys are normalized in this order: strip a leading
`module.`, apply your `keywords → replacement` rewrite (only where the bare key
*starts with* `keywords`), then re-add `module.` when `world_size > 1`.

```python
# in cfg.hooks of the fine-tune config
dict(
    type="CheckpointLoader",
    keywords="module.student.backbone",
    replacement="module.backbone",
),
```

:::{important}
A remap that matches **zero** parameters raises, rather than silently training
from random init. If you see `No weight found` / `Missing keys: [...everything]`,
your `keywords` prefix is wrong. Judge a successful load by the loss curves of
the *new* head (`loss_cls`/`dice`), not just the absence of errors.
:::

For partial / programmatic warm starts (loading only a submodule), use the
lower-level {py:func}`~pimm.export.load_pretrained` helper from `pimm.export` — see
{doc}`../models/index`.

## Resume on a different number of GPUs

The default `standard`/DCP format reshards automatically — change the resource
flag and resume:

```bash
# 8-GPU run, resumed on 4 GPUs:
pimm submit --site s3df --resources.nnodes 1 --resources.nproc-per-node 4 \
  --train.config <cfg> --run.name my-run --train.resume
```

This works because the trainer state is a Distributed Checkpoint and the global
batch size is fixed, so iterations-per-epoch is identical. With the **legacy**
single-file format, resharding is not automatic — set
`resume_strict_state=False` to skip the world-size assertion. Full details and
the strict-mode escape hatch are in {doc}`../checkpoints/resume_world_size`.

## Chained (requeued) runs

A requeue chain resumes automatically: attempt 1 runs fresh, attempts 2+ resume
from the newest complete checkpoint in the same experiment directory. See
{doc}`chaining`.

## Troubleshooting

:::{dropdown} "It restarted from scratch instead of resuming"
Check that a *complete* checkpoint exists
(`python -m pimm.utils.path latest-checkpoint exp/.../run`). A `standard`/DCP
checkpoint needs its `.complete` marker; an interrupted save leaves only the
`.tmp` dir and the previous good `.prev`.
:::

:::{dropdown} "My config change didn't take effect on resume"
Resume uses the saved `exp/.../config.py`. To change behavior, either start a
new run, or edit the saved artifact directly (advanced) — but prefer a new run
with a child config for anything you want to track.
:::

:::{dropdown} "A custom dataset/hook isn't found on resume"
Resume loads the *dumped* config, which has no config-level `__import__`. Custom
datasets/hooks must be registered in the package `__init__.py` so their
registration side effects run regardless. See {doc}`../getting_started/concepts`.
:::

:::{dropdown} "World-size mismatch error on resume"
Strict resume (`resume_strict_state=True`) refuses to remap per-rank
dataloader/RNG state saved under a different world size. Use the DCP/standard
format to reshard, or set `resume_strict_state=False` for the legacy format.
:::

## Next

- {doc}`../checkpoints/index` — formats, atomicity, and the DCP layout.
- {doc}`../models/index` — warm-start helpers and `hf://` weights.
