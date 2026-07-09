# Resuming on Slurm

The mechanics of resuming - what's restored, mid-epoch resume, fine-tune
warm-start, and resharding across a world-size change - are general and live in
{doc}`../checkpoints/resuming` and {doc}`../checkpoints/saving_and_loading`. Two things are **specific to managed Slurm runs**: resuming a submitted job
and automatic requeue chains.

## Resume a submitted run

Same `--train.resume` flag as a local run, through `pimm submit`:

```bash
pimm submit --site mycluster \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask \
  --run.name my-run \
  --train.resume
```

Resume reuses the saved `exp/<group>/<name>/config.py` and the code snapshot
under `exp/<group>/<name>/code/`, and picks the newest *complete* checkpoint in
the experiment directory. If none is complete, it exits rather than restarting.
(Full details: {doc}`../checkpoints/resuming`.)

Change the resource flags on the resume and the trainer state reshards automatically:

```bash
# 8-GPU run, resumed on 4 GPUs:
pimm submit --site mycluster --resources.nnodes 1 --resources.nproc-per-node 4 \
  --train.config <cfg> --run.name my-run --train.resume
```

## Chained (requeued) runs

Requeue chains resume automatically; see {doc}`chaining` for the mechanics.

:::{dropdown} "It restarted from scratch instead of resuming"
Check that a *complete* checkpoint exists
(`python -m pimm.utils.path latest-checkpoint exp/.../run`). A `standard`/DCP
checkpoint needs its `.complete` marker; an interrupted save leaves only the
`.tmp` dir and the previous good `.prev`.
:::
