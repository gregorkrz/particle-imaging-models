# Checkpoint hooks

Saving is done by **hooks** in `cfg.hooks`. They delegate all save/load
semantics to {py:class}`~pimm.utils.checkpoints.CheckpointManager` (`pimm/utils/checkpoints.py`); the hook classes
just decide *when*. There are three, registered in
`pimm/engines/hooks/checkpoint.py`.

```{list-table}
:header-rows: 1
:widths: 26 74

* - Hook
  - Role
* - `CheckpointLoader`
  - Loads weights and (when `resume=True`) restores full training state in
    `before_train()`. Supports key remapping for fine-tuning.
* - `CheckpointSaver`
  - Evaluator-aware saver. Writes on a configured save step or eval step and in
    `after_train()`; writes `model_best.pth` when the metric improves.
* - `CheckpointSaverIteration`
  - Long-run iteration saver. Writes every `save_freq` steps and in
    `after_train()`.
```

:::{important}
**Hook order matters.** A saver can only mark a *best* checkpoint after an
evaluator has written `trainer.comm_info["current_metric_value"]`. Put evaluator
hooks **before** the saver in the list.
:::

## Save by cadence

::::{tab-set}

:::{tab-item} Epoch / eval-step cadence
```python
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator", every_n_steps=1000, write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=None, evaluator_every_n_steps=1000),
    dict(type="FinalEvaluator", test_last=False),
]
```
{py:class}`~pimm.engines.hooks.checkpoint.CheckpointSaver` writes the rolling checkpoint and, when the evaluator metric
improves, `model_best.pth`. `save_freq=None` means "save at eval points / end of
training"; set an integer for periodic `iter_<step>.pth` snapshots.
:::

:::{tab-item} Iteration cadence (long runs)
```python
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaverIteration", save_freq=1000),
]
```
{py:class}`~pimm.engines.hooks.checkpoint.CheckpointSaverIteration` writes a fresh rolling checkpoint every `save_freq`
optimizer steps and once more in `after_train()`. This is the right saver for
walltime-limited HPC runs and requeue chains — pick a `save_freq` small enough
that an attempt always leaves a recent complete checkpoint before it times out.
:::

::::

`CheckpointSaverIteration.__init__(save_freq=None, evaluator_every_n_steps=None)`;
the older `backend=` argument is a deprecated alias for `checkpoint_format`
(`dcp`→`standard`, `torch`→`legacy`).

:::{note}
Both savers build the payload with per-rank RNG and dataloader state, so they
**must be called on every rank** — the `standard`-format save is a collective DCP
operation. Rank-0-only side artifacts (`model_best.pth`, `iter_<step>.pth`) are
guarded internally.
:::

## Loading & fine-tune remap

`CheckpointLoader.__init__(keywords="", replacement=None, replacements=None,
strict=False)`. When `cfg.resume=False` it loads model weights only; when
`cfg.resume=True` it restores the full training state.

For fine-tuning from a checkpoint whose keys don't match, remap them:

```python
dict(
    type="CheckpointLoader",
    keywords="module.student.backbone",   # match keys starting with this…
    replacement="module.backbone",        # …rewrite the prefix to this
),
```

Use `replacements=[...]` for multiple independent rewrites. A remap that matches
**zero** parameters raises rather than silently random-initializing. See
{doc}`../hpc/resuming` for the full key-normalization order.

## Writing a periodic side artifact

If you want a custom save cadence (e.g. dumping diagnostics), write a small hook
and place it after the saver. The generic checkpoint payload saves model,
optimizer, scheduler, scaler, dataloader, RNG, and trainer state — but **not**
arbitrary hook attributes. State a hook must resume needs its own contract.

```python
from pimm.engines.hooks.builder import HOOKS
from pimm.engines.hooks.default import HookBase
from pimm.utils.comm import is_main_process

@HOOKS.register_module()
class EveryNStepsDump(HookBase):
    def __init__(self, every=1000):
        self.every = int(every)
        self.n = 0
    def before_train(self):
        self.n = int(getattr(self.trainer, "global_step", 0) or 0)
    def after_step(self):
        self.n += 1
        if self.n % self.every == 0 and is_main_process():
            ...  # write your artifact under self.trainer.cfg.save_path
```

Remember to import new hook modules from `pimm/engines/hooks/__init__.py` so
their registration runs (and survives a resume). See {doc}`../hooks/index`.

## Next

- {doc}`huggingface` — the {py:class}`~pimm.engines.hooks.export.PushToHub` hook for Hub uploads during training.
- {doc}`export` — consolidating a checkpoint into a portable artifact.
