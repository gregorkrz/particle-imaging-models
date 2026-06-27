# Hooks

Hooks are how pimm keeps the trainer generic. The `DefaultTrainer` loop only
knows how to move a batch to the device, call `model(input_dict)`, read a scalar
`loss`, and step the optimizer. *Everything else* — timing, logging,
diagnostics, periodic validation, checkpointing, Hugging Face upload — is a
**hook** plugged into the training lifecycle.

A hook is a small object with lifecycle methods (`before_train`, `after_step`,
…). You list hooks as config dicts in `cfg.hooks`; the trainer builds each one
through the `HOOKS` registry and calls the matching method at the right moment.

```python
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator", every_n_steps=1000),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="FinalEvaluator", test_last=False),
]
```

:::{seealso}
New to pimm? Read {doc}`../getting_started/concepts` first — hooks build on the
registry idea (§2) and the one-forward-contract trainer (§4).
:::

## The lifecycle

`TrainerBase.register_hooks()` builds each dict through `HOOKS`, asserts the
result is a {py:class}`~pimm.engines.hooks.default.HookBase`, and attaches a **weak proxy to the trainer** at
`hook.trainer`. Hooks then fire in config order at fixed points in
`Trainer.train()`:

```text
register_hooks()           ── build cfg.hooks via HOOKS registry
modify_config(cfg)         ── late config edits (before the writer is built)
build writer (rank 0)
before_train()             ── CheckpointLoader loads weights / restores resume
  for epoch in start_epoch .. max_epoch-1:
    before_epoch()
      for batch in train_loader:
        before_step()
        run_step()         ── model(input_dict) → loss → backward → opt/sched
        _record_step_state()   ← resume position captured HERE
        after_step()       ── timers, scalar logging, evaluators, savers
    after_epoch()          ── epoch averages, optional CUDA cache empty
after_train()              ── final eval + final save, then writer closes
```

:::{important}
**Hook order matters.** Hooks fire in the order they appear in `cfg.hooks`. The
canonical rule: put **evaluators before savers**, so a {py:class}`~pimm.engines.hooks.checkpoint.CheckpointSaver` sees
the `current_metric_value` an evaluator just wrote and can mark
`model_best.pth`. See {doc}`../checkpoints/saving_and_loading`.
:::

:::{note}
`_record_step_state()` runs **before** `after_step()`. A saver in `after_step()`
therefore records the resume position *after* the just-completed batch — exactly
what you want for mid-epoch resume. A hook with its own step counter should seed
it from `trainer.global_step` in `before_train()` so it stays resume-aware.
:::

### Lifecycle methods

```{list-table}
:header-rows: 1
:widths: 24 76

* - Method
  - When it runs / typical use
* - `modify_config(cfg)`
  - After all hooks are registered, before the writer is built. Late config
    edits only — e.g. `WandbNamer` derives `wandb_run_name` here. Not forwarded
    by `ModelHook`.
* - `before_train()`
  - Once, before the first epoch. Checkpoint loading/resume; registering
    forward hooks; seeding resume-aware counters from `trainer.global_step`.
* - `before_epoch()` / `after_epoch()`
  - Wrap each epoch. Epoch-cadence evaluators (`every_n_steps == 0`) run from
    `after_epoch()`; `InformationWriter` flushes epoch averages here.
* - `before_step()` / `after_step()`
  - Wrap each optimizer step. Per-step timing, scalar logging, step-cadence
    evaluators (`every_n_steps > 0`), and checkpoint savers live here.
* - `after_train()`
  - Once, after a distributed barrier. `FinalEvaluator` runs the held-out test;
    final checkpoint is written. The writer is closed after hooks finish.
```

## How the trainer and hooks talk

Hooks read and write a few well-known channels on `hook.trainer`:

```{list-table}
:header-rows: 1
:widths: 32 68

* - Channel
  - Contents
* - `trainer.comm_info`
  - Per-step exchange dict. Keys: `epoch`, `iter`, `iter_per_epoch`,
    `input_dict`, `model_output_dict`, and the evaluator outputs
    `current_metric_value` / `current_metric_name`.
* - `trainer.storage`
  - In-process scalar histories (`EventStorage`). Put scalars here when
    `InformationWriter` or epoch averages should pick them up.
* - `trainer.writer`
  - TensorBoard / W&B writer, built on rank 0 only (`None` elsewhere). Call
    `add_scalar(tag, value, step)` for explicit logging keys.
* - `trainer.global_step` / `start_epoch` / `start_iter`
  - Progress counters; seed resume-aware schedules from these.
* - `trainer.model`
  - The (possibly DDP-wrapped) model. Unwrap `.module` before touching model
    internals.
```

{py:class}`~pimm.engines.hooks.default.ModelHook` is a special bridge: when the *model itself* behaves like a
`HookBase` (e.g. an SSL model that needs per-step EMA updates), `ModelHook`
unwraps DDP, hands the model the same trainer reference, and forwards
`before_train`/`before_epoch`/`before_step`/`after_step`/`after_epoch`/`after_train`
to it. It does **not** forward `modify_config`.

## The hook families

- {doc}`Logging <logging>` — run naming, iteration timing, and scalar/console logging — `WandbNamer`, `IterationTimer`, `InformationWriter`, and the writer / storage / `comm_info` channels they use.
- {doc}`Diagnostics <diagnostics>` — health monitors and runtime mutators — gradient norms, prototype usage, feature std, resources, parameter counts, plus weight-decay / dtype / annealing / profiling hooks.
- {doc}`Evaluation <../evaluation/index>` — in-loop evaluators that write `current_metric_value`, SSL probe suites, and final held-out testing.
- {doc}`Checkpointing <../checkpoints/saving_and_loading>` — `CheckpointLoader`, `CheckpointSaver`, `CheckpointSaverIteration` — when to save, fine-tune remapping, and `model_best.pth` selection.
- {doc}`Hugging Face <../checkpoints/huggingface>` — the `PushToHub` hook for uploading exported weights to the Hub during training.

## Registering a hook

A hook becomes buildable once its `@HOOKS.register_module()` decorator runs, so a
new hook file must be imported in `pimm/engines/hooks/__init__.py`.

## Next

- {doc}`logging` — the everyday logging hooks.
- {doc}`diagnostics` — model-health monitors and runtime mutators.
- {doc}`../evaluation/index` — evaluators, probe suites, and final testing.

```{toctree}
:hidden:

logging
diagnostics
```
