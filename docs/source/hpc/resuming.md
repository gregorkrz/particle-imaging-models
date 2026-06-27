# Resuming on Slurm

The mechanics of resuming — what's restored, mid-epoch resume, fine-tune
warm-start, and resharding across a world-size change — are general and live in
{doc}`../checkpoints/resuming` and {doc}`../checkpoints/saving_and_loading`. This
page covers what's **specific to managed Slurm runs**: resuming a submitted job
and automatic requeue chains.

## Resume a submitted run

Same `--train.resume` flag as a local run, through `pimm submit`:

```bash
pimm submit --site s3df \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask \
  --run.name my-run \
  --train.resume
```

Resume reuses the saved `exp/<group>/<name>/config.py` and the code snapshot
under `exp/<group>/<name>/code/`, and picks the newest *complete* checkpoint in
the experiment directory. If none is complete, it exits rather than restarting.
(Full details: {doc}`../checkpoints/resuming`.)

## Resume on a different number of GPUs

The default `standard`/DCP format reshards automatically — change the resource
flags and resume:

```bash
# 8-GPU run, resumed on 4 GPUs:
pimm submit --site s3df --resources.nnodes 1 --resources.nproc-per-node 4 \
  --train.config <cfg> --run.name my-run --train.resume
```

This is safe because the trainer state is a Distributed Checkpoint and the global
batch size is fixed. The legacy single-file format needs
`resume_strict_state=False`. See {doc}`../checkpoints/resuming` for the full
reshard / strict-mode discussion.

## Chained (requeued) runs

A requeue chain resumes automatically: attempt 1 runs fresh, attempts 2+ resume
from the newest complete checkpoint in the same experiment directory. Pair this
with {py:class}`~pimm.engines.hooks.checkpoint.CheckpointSaverIteration` and a
`save_freq` small enough that every attempt leaves a recent complete checkpoint
before it times out. See {doc}`chaining`.

:::{dropdown} "It restarted from scratch instead of resuming"
Check that a *complete* checkpoint exists
(`python -m pimm.utils.path latest-checkpoint exp/.../run`). A `standard`/DCP
checkpoint needs its `.complete` marker; an interrupted save leaves only the
`.tmp` dir and the previous good `.prev`.
:::

## Next

- {doc}`../checkpoints/resuming` — what's restored, mid-epoch, and resharding.
- {doc}`../checkpoints/saving_and_loading` — fine-tune warm-start and key remapping.
- {doc}`chaining` — requeue chains, walltime, QOS, and accounts.
