# Resuming across a world-size change

You started an 8-GPU run; now you only have 4 GPUs (or you want to scale *up*).
What happens depends on the checkpoint format.

## The default format reshards automatically

With the `standard`/DCP format (the default, and what the launcher selects for
multi-rank/requeued/FSDP2 runs), just change the resource flag and resume:

```bash
# Started on 8 GPUs; resume on 4 — no extra flags.
pimm submit --site s3df \
  --resources.nnodes 1 --resources.nproc-per-node 4 \
  --train.config <cfg> --run.name my-run --train.resume
```

This works because:

- The trainer state lives in a **Distributed Checkpoint** (`trainer.dcp/`), which
  resharded reads/writes natively — model, optimizer, scheduler, step,
  samples-seen, and best-metric all redistribute cleanly.
- The **global batch size is fixed**, so iterations-per-epoch is identical
  regardless of GPU count. Your schedule and accounting stay aligned.

## Legacy format: the escape hatch

The legacy single-file format does **not** reshard. Strict resume
(`resume_strict_state=True`, the default) refuses to remap per-rank dataloader /
RNG state saved under a different world size, and raises. To resume anyway:

```bash
pimm launch --train.config <cfg> --run.name my-run --train.resume \
  -- resume_strict_state=False
```

What `resume_strict_state=False` does, concretely:

- skips the `world_size` assertion on the saved distributed dataloader / RNG
  state, and
- you additionally skip the dataloader cursor (torchdata asserts lazily on first
  `__iter__`).

Model / optimizer / scheduler / step state reshard fine because they are not
rank-partitioned in the single-file payload. This is safe specifically because
the global batch size is fixed, so iters/epoch is identical — only the per-rank
RNG/dataloader bookkeeping is being relaxed.

:::{warning}
`resume_strict_state=False` trades *exact* data/RNG resume for the ability to
change GPU count with the legacy format. The continued trajectory is no longer
bitwise-identical to an uninterrupted run. Prefer the `standard`/DCP format when
you anticipate changing world size.
:::

## Which format am I using?

```text
model/last/trainer.dcp/   present  →  standard / DCP  (reshards automatically)
model/model_last.pth      present  →  legacy          (needs resume_strict_state=False)
```

Force the reshardable format for new runs:

```bash
pimm launch --train.config <cfg> -- checkpoint_format=standard
```

(The launcher already does this for multi-rank, requeued, and FSDP2 runs.)

## See also

- {doc}`index` — the two checkpoint formats in full.
- {doc}`../distributed/index` — why fixed global batch size makes this safe.
- {doc}`../hpc/resuming` — the broader resume vs warm-start picture.
