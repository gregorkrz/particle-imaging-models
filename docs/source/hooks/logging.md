# Logging hooks

Logging hooks turn what happens inside the training loop into three things you
can read: a **console line** per step, **scalar histories** for epoch averages,
and **TensorBoard / W&B** curves. They live in `pimm/engines/hooks/logging.py`.

Before the individual hooks, it helps to know the three channels they write to.

## The three logging channels

```{list-table}
:header-rows: 1
:widths: 26 38 36

* - Channel
  - What it is
  - Who reads it
* - `trainer.storage`
  - In-process scalar histories (`EventStorage`); `put_scalar(name, value)`.
  - `InformationWriter` (epoch averages), other hooks.
* - `trainer.comm_info["iter_info"]`
  - A string assembled per step for the console.
  - The logger that prints the one-line status.
* - `trainer.writer`
  - TensorBoard / W&B writer; `add_scalar(tag, value, step)`. Rank 0 only
    (`None` on other ranks).
  - TensorBoard / Weights & Biases dashboards.
```

The usual division of labour: write a scalar to `trainer.storage` when you want
{py:class}`~pimm.engines.hooks.logging.InformationWriter` to average it over an epoch and print it; write directly to
`trainer.writer` when you want an explicit named curve in TensorBoard/W&B.

:::{note}
`trainer.writer` is built on the **main process only**. Always guard direct
writes with `is_main_process()` (and check the writer is not `None`). See
{doc}`writing_hooks`.
:::

A representative logging stack:

```python
hooks = [
    dict(type="WandbNamer", keys=["model.backbone.type", "optimizer.lr"]),
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", log_frequency=20),
    # ... evaluators, savers ...
]
```

## WandbNamer

Builds a human-readable `wandb_run_name` from config values, so runs in the W&B
UI are named by what actually varies between them rather than by a random id.

`WandbNamer(keys=(), sep="-", format_numbers=True, extra=None)`

- Runs in `modify_config(cfg)` — *before* the writer is constructed, which is the
  only place a run name can still be set.
- `keys` are dotted config paths (e.g. `"model.backbone.type"`,
  `"optimizer.lr"`); their resolved values are joined with `sep`.
- `format_numbers` tidies numeric values; `extra` appends fixed tokens.
- It **defers to the CLI**: if a run name was already set on the command line,
  {py:class}`~pimm.engines.hooks.logging.WandbNamer` leaves it alone.

:::{tip}
`WandbNamer` is the only common hook whose work happens in `modify_config`. Put
it early in `cfg.hooks`; order among `modify_config`-only hooks is otherwise
irrelevant since they all run before the writer exists.
:::

## IterationTimer

Measures where wall-clock time goes and projects when the run will finish.

`IterationTimer(warmup_iter=1)`

- Writes `data_time` (time spent waiting on the dataloader) and `batch_time`
  (full step time) into `trainer.storage`, plus an estimated **remaining time**
  into `comm_info["iter_info"]`.
- `warmup_iter` skips the first few steps from the timing average — the first
  iterations are dominated by dataloader spin-up, lazy CUDA init, and cudnn
  autotuning and would otherwise poison the ETA.

:::{tip}
A large `data_time` relative to `batch_time` means the GPU is starving on input.
Raise `num_worker`, enable persistent workers, or simplify transforms before
reaching for a bigger model.
:::

## InformationWriter

The workhorse. It collects per-step scalars the model emits and renders both the
per-step console line and the end-of-epoch averages.

`InformationWriter(log_frequency=1, step_offset=0)`

- Filters `comm_info["model_output_dict"]` down to **scalar-like** entries and
  excludes large tensors, so dumping a `(N, C)` tensor into the output dict won't
  flood the log.
- Maps `total_loss` to the displayed `loss` when present — models that compute a
  weighted sum into `total_loss` get that shown as the headline loss (see the
  output-key table in {doc}`../getting_started/concepts`).
- Logs the filtered train-batch scalars every `log_frequency` steps and writes
  **epoch averages** in `after_epoch()`.
- `step_offset` shifts the logged step index (useful when stitching logs across
  resumes).

What lands where:

```text
model forward → output_dict {loss, total_loss, loss_cls, dice, acc, ...}
                                    │
                  InformationWriter filters to scalars
                       │                         │
              console line              trainer.writer.add_scalar(...)
        comm_info["iter_info"]            (TensorBoard / W&B curves)
                                                 │
                          after_epoch(): epoch averages from trainer.storage
```

:::{seealso}
Want a scalar to show up here? Return it in your model's output dict as a Python
float or 0-d tensor, or `put_scalar` it into `trainer.storage`. Tensors with
shape are dropped by the scalar filter on purpose.
:::

## Putting it together

```python
hooks = [
    dict(type="WandbNamer", keys=["model.backbone.type", "scheduler.lr"], sep="_"),
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", log_frequency=50),
    dict(type="SemSegEvaluator", every_n_steps=2000),
    dict(type="CheckpointSaver", save_freq=None),
]
```

Console output then looks roughly like:

```text
Train: [3/50][120/1563] data_time: 0.01 batch_time: 0.42 remain: 02:13:44 loss: 0.731 loss_cls: 0.402 dice: 0.329
```

## Next

- {doc}`diagnostics` — gradient norms, feature std, resources, and runtime
  mutators.
- {doc}`writing_hooks` — emit your own scalars and curves from a custom hook.
- {doc}`../evaluation/index` — evaluators that turn validation into a selection
  metric.
