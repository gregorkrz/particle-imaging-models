# Checkpoints

pimm uses **one checkpoint format for every parallelism** — single-GPU,
multi-GPU, and multi-node all write the same thing — so resume is predictable
regardless of how many devices you used. Checkpoints are atomic, capture the
*full* training state, and (in the default format) reshard across world sizes.

- {doc}`Saving & loading <saving_and_loading>` — saver/loader hooks, cadence, fine-tune key remapping, programmatic loading.
- {doc}`Resuming <resuming>` — exact resume, mid-epoch, and resharding across a world-size change.
- {doc}`Exporting <exporting>` — `pimm export`, `save_pretrained`, portable weights.
- {doc}`Hugging Face <huggingface>` — auto-push during training, `hf://` fine-tune.

## What's in a checkpoint

A pimm training checkpoint is a versioned payload (schema version 3) that records
everything needed to *continue exactly*, not just the weights:

```python
{
  "schema": "pimm.trainer_checkpoint", "version": 3,
  "model":     {"state_dict": ...},
  "optimizer": {"state_dict": ..., "class": ...},
  "scheduler": {"state_dict": ..., "class": ..., "total_steps": ...},
  "scaler":    {"enabled": ..., "state_dict": ...},
  "dataloader":{"backend": ..., "state": ..., "world_size": ..., ...},
  "rng":       {"world_size": ..., "state": [...per-rank...]},
  "trainer":   {"epoch": ..., "iter_in_epoch": ..., "global_step": ...,
                "samples_seen": ..., "best_metric_value": ...},
  "logger": ..., "distributed": ...,
}
```

This is what makes {doc}`exact resume <resuming>` possible — RNG (Python /
NumPy / CPU / all CUDA), the stateful dataloader position, the step counter, and
samples-seen all travel with the optimizer state. Helper functions in
`pimm/utils/checkpoints.py` read both v3 and older flat layouts.

## The two on-disk formats

The layout is chosen by `cfg.checkpoint_format`.

::::{tab-set}

:::{tab-item} standard (default)
A **split directory** — portable weights next to a Distributed Checkpoint for
everything else:

```text
exp/<dataset>/<name>/model/
  last/                 # resume from here
    weights.pth         # portable model weights — plain torch.load(...)["state_dict"]
    trainer.dcp/        # optimizer / scheduler / RNG / dataloader as a DCP
    .complete           # written last; marks the checkpoint atomically complete
  last.prev/            # previous complete checkpoint (rotated)
  model_best.pth        # best-metric model weights only (for eval / export)
```

- **Portable weights, always.** `last/weights.pth` and `model_best.pth` are
  ordinary single-file state dicts — load them anywhere without DCP.
- **Reshards automatically.** The DCP `trainer.dcp/` lets you resume on a
  different number of GPUs/nodes with no extra flags.
- This is *not* a pure-DCP directory — it is a split checkpoint that wraps one.
:::

:::{tab-item} legacy
A single monolithic file (model + trainer state together):

```text
exp/<dataset>/<name>/model/
  model_last.pth        # everything in one file
  model_last.pth.prev   # previous complete checkpoint
  model_best.pth        # best-metric copy
```

Simple and dependency-free, but it does **not** reshard across world sizes
(resume on a different GPU count needs `resume_strict_state=False`). Select it
with:

```bash
pimm launch --train.config <cfg> --run.name <name> -- checkpoint_format=legacy
```
:::

::::

:::{note}
The deprecated `backend` alias maps `backend="dcp"` → `standard` and
`backend="torch"` → `legacy`; the top-level `checkpoint_format` key takes
precedence. The launcher already defaults to the reshardable `standard`/DCP
format for multi-rank, requeued, or FSDP2 runs — you rarely set this by hand.
:::

## Atomic by construction

A save never corrupts the previous checkpoint, because every publish goes
through a temp + rotate + rename:

```text
write → <target>.tmp        (fsync / collective DCP write)
rotate → <target> → <target>.prev
rename → <target>.tmp → <target>
mark  → .complete           (directory formats)
```

For single-file saves, `atomic_torch_save()` writes `<file>.tmp`, fsyncs,
rotates the old file to `<file>.prev`, then atomically replaces `<file>`. For DCP
directories, rank 0 writes the `.complete` marker, rotates `.prev`, and swaps the
directory, with all ranks synchronizing around the rename.

A directory checkpoint counts as **complete only if it exists and contains
`.complete`**. The launcher's resume picks the newest complete checkpoint among
`last`, `last.prev`, `model_last.pth`:

```bash
python -m pimm.utils.path latest-checkpoint exp/panda/pretrain/my-run
```

## Deterministic checkpointing

"Deterministic" here means: resume produces the **same continued trajectory**
you would have gotten without the interruption. Because RNG state (all streams,
per rank), the dataloader's exact position, and the optimizer/scheduler state are
all captured, a resumed run does not re-see data, re-roll augmentations, or jump
the LR schedule. Combined with `seed` and `deterministic=True`, this gives
reproducible long runs across timeouts and requeues.

See {doc}`resuming` for the resume mechanics, including changing GPU count.

## Quick reference

```{list-table}
:header-rows: 1
:widths: 34 66

* - File
  - What it is
* - `model/last/weights.pth`
  - portable model weights (standard format)
* - `model/last/trainer.dcp/`
  - optimizer/scheduler/RNG/dataloader as a reshardable DCP
* - `model/last/.complete`
  - atomicity marker — without it the checkpoint is ignored
* - `model/last.prev/`
  - previous complete checkpoint (rotation)
* - `model/model_best.pth`
  - best-metric model weights only (eval / export)
* - `model/model_last.pth`
  - legacy single-file checkpoint (legacy format only)
```

## Next

- {doc}`saving_and_loading` — control *when* checkpoints are written, and how to load them.
- {doc}`resuming` — continue an interrupted run, including on a different GPU count.
- {doc}`exporting` — turn a checkpoint into a portable pretrained artifact.
- {doc}`huggingface` — push/pull checkpoints to the Hub.

```{toctree}
:hidden:

saving_and_loading
resuming
exporting
huggingface
```
