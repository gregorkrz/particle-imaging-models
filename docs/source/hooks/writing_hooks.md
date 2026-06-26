# Writing a hook

A hook is a subclass of {py:class}`~pimm.engines.hooks.default.HookBase` with one or more lifecycle methods. The
trainer builds it from a config dict, attaches itself at `self.trainer`, and
calls your methods at the right point in the loop. This page covers the minimal
recipe, the resume-aware gotchas, and the do's and don'ts.

:::{seealso}
Read {doc}`index` for the full lifecycle diagram and the `comm_info` / `storage`
/ `writer` channels referenced below.
:::

## The recipe

1. **Subclass `HookBase`** (`pimm/engines/hooks/default.py`).
2. **Register it** with `@HOOKS.register_module()`.
3. Keep side effects in the **narrowest lifecycle method** that fits the job.
4. **Import the module from `pimm/engines/hooks/__init__.py`** if it is a new
   file, so the registration runs before config construction *and survives
   resume*.
5. Add the hook dict to `cfg.hooks`.

### Example: a periodic scalar hook

```python
from pimm.engines.hooks.builder import HOOKS
from pimm.engines.hooks.default import HookBase
from pimm.utils.comm import is_main_process


@HOOKS.register_module()
class MyScalarHook(HookBase):
    def __init__(self, log_frequency=100):
        self.log_frequency = int(log_frequency)
        self.step_count = 0

    def before_train(self):
        # Seed from trainer.global_step so the counter is resume-aware.
        self.step_count = int(getattr(self.trainer, "global_step", 0) or 0)

    def after_step(self):
        self.step_count += 1
        if self.step_count % self.log_frequency != 0:
            return
        if is_main_process() and self.trainer.writer is not None:
            self.trainer.writer.add_scalar("my/value", 1.0, self.step_count)
```

Config:

```python
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="MyScalarHook", log_frequency=100),
    dict(type="CheckpointSaverIteration", save_freq=1000),
]
```

## Why the `__init__.py` import is non-negotiable

Registration is a **decorator side effect**: `@HOOKS.register_module()` only runs
when the module is imported. A config can `__import__` a module to register it
for a fresh run — but a **resume reloads the dumped config**, which has no
`__import__` line. The class is then unregistered and the resume crashes.

:::{important}
Always import new hook modules from `pimm/engines/hooks/__init__.py`. This is the
same gotcha that applies to models, datasets, and transforms — see
{doc}`../getting_started/concepts` (§2) and {doc}`../hpc/resuming`.
:::

## Resume-aware counters

The trainer records the checkpointable step state in `_record_step_state()`
**before** `after_step()` runs. So a hook acting in `after_step()` already sees
the resume position for the just-completed batch.

But the trainer does **not** restore your hook's own attributes. If your hook
keeps a step counter, seed it in `before_train()`:

```python
def before_train(self):
    self.step_count = int(getattr(self.trainer, "global_step", 0) or 0)
```

For a mid-epoch resume you can also reconstruct from
`trainer.start_epoch * len(trainer.train_loader) + trainer.start_iter`. Without
this, a resumed run restarts your counter at 0 and your cadence drifts out of
alignment with the global step.

## The hook state contract

The generic checkpoint payload saves **model, optimizer, scheduler, scaler,
dataloader, RNG, and trainer state** — and nothing else. Arbitrary hook
attributes are **not** persisted.

```{list-table}
:header-rows: 1
:widths: 50 50

* - Saved automatically
  - You must persist yourself
* - model / optimizer / scheduler / scaler
  - any counter, accumulator, or buffer on the hook
* - dataloader position, RNG, trainer progress
  - probe weights & their optimizer (e.g. `OnlineLinearProbe`)
* - `best_metric_value`
  - registered forward-hook handles (re-register in `before_train`)
```

If a counter can be **derived** from `trainer.global_step`, derive it in
`before_train()` (cheap, no new format). If a hook owns genuine state that must
resume bit-exactly, give it an **explicit checkpoint contract**: write a side
file under `cfg.save_path` on save, and read it back in `before_train()`.

:::{warning}
{py:class}`~pimm.engines.hooks.eval.pretrain.online_probe.OnlineLinearProbe` is the cautionary example: its probe head and optimizer live
on the hook, not on `trainer.model`, so the checkpoint payload does **not** save
them. A resumed probe starts from scratch. That's fine for a diagnostic probe;
it would not be fine for anything load-bearing.
:::

## Distributed correctness

```{list-table}
:header-rows: 1
:widths: 36 64

* - Situation
  - Do this
* - Single-writer side effect (log, file write)
  - Guard with `is_main_process()`; `trainer.writer` is `None` off rank 0.
* - Other ranks must wait for rank 0
  - Call `synchronize()` so ranks don't race ahead or deadlock a barrier.
* - Collective op (DCP save, all-gather)
  - Run it on **every** rank; guard only the rank-0 side artifacts internally.
```

## Choosing where scalars go

```{list-table}
:header-rows: 1
:widths: 40 60

* - Goal
  - Channel
* - Show up in epoch averages / `InformationWriter`
  - `trainer.storage` (`put_scalar`)
* - Explicit named TensorBoard / W&B curve
  - `trainer.writer.add_scalar(tag, value, step)` (rank 0)
* - One-line console status
  - append to `trainer.comm_info["iter_info"]`
```

## Do's and don'ts

**Do**

- Respect hook order: if a saver needs a metric, put the evaluator first.
- Pick the narrowest lifecycle method; register/remove PyTorch forward hooks in
  `before_train()` / `after_train()`.
- Unwrap DDP (`model.module`) before touching model internals.
- Seed counters from `trainer.global_step` for resume-awareness.

**Don't**

- Don't assume `trainer.writer` exists — it's rank-0 only and may be `None`.
- Don't rely on the generic payload to save hook attributes; it won't.
- Don't register a hook only via a config `__import__`; it dies on resume.
- Don't run collective saves/gathers under an `is_main_process()` guard — that
  deadlocks the other ranks.
- Don't `cp` a warm-start file into a live run's `model/last/`; the next atomic
  save wipes extra files there.

## Next

- {doc}`logging` and {doc}`diagnostics` — patterns to copy from the built-in
  hooks.
- {doc}`../checkpoints/hooks` — savers, loaders, and warm-start remapping.
- {doc}`../evaluation/index` — writing an evaluator that drives `model_best`.
