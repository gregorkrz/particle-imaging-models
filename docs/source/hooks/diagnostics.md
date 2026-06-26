# Diagnostics & runtime hooks

Diagnostic hooks watch the *health* of training — gradients, feature collapse,
prototype usage, hardware utilization — and the **mutating** hooks change
runtime behavior such as weight decay, parameter dtype, and attention-mask
annealing. They live in `pimm/engines/hooks/diagnostics.py`,
`resources.py`, `optimizer.py`, and `profiling.py`.

These split into two groups: **read-only monitors** that only log, and
**mutators** that change optimizer groups, dtypes, or the model's anneal state.

## Read-only monitors

### GradientNormLogger

Logs the global (and optionally per-layer) gradient norm — the first thing to
check when a loss diverges or plateaus.

`GradientNormLogger(norm_type=2.0, log_per_layer=False, log_frequency=1, prefix="grad_norm")`

- Computes the `norm_type`-norm of gradients after backward.
- `log_per_layer=True` breaks the norm down by parameter group/layer under
  `prefix` — useful for spotting a single layer that explodes or dies.

:::{tip}
A gradient norm that spikes right at the end of an SSL warmup usually means the
*objective* got too hard, not that the LR is wrong — lower the mask ratio/size
base or lengthen its warmup before touching the LR.
:::

### PrototypeUsageLogger

For prototype-based SSL (Sonata-style heads), logs how the prototype assignments
are distributed — the canonical detector for **prototype collapse**, where the
model routes everything to a handful of prototypes.

`PrototypeUsageLogger(log_frequency=10, prefix="prototypes")`

- Registers PyTorch **forward hooks** on the model's prototype heads in
  `before_train()` (unwrapping DDP first) and **removes them in `after_train()`**.
- Logs usage entropy / per-prototype occupancy under `prefix`.

### FeatureStdMonitor

Tracks the standard deviation of student and teacher features — a near-zero std
is the textbook signature of representation collapse in self-distillation.

`FeatureStdMonitor(log_frequency=10, prefix="feature_std", monitor_student=True, monitor_teacher=True, track_channels=False)`

- Registers forward hooks on the Sonata student/teacher modules; removed at
  `after_train()`.
- `track_channels=True` additionally reports per-channel std.

### ResourceUtilizationLogger

Logs CPU/RAM and GPU memory so you can right-size jobs and catch leaks.

`ResourceUtilizationLogger(log_frequency=10, prefix="resources", log_per_gpu=False, log_cpu=True, log_system_memory=True, per_process=True)`

- Uses `psutil` and `pynvml` when available; falls back to `torch.cuda` memory
  metrics otherwise.
- `per_process=True` reports this process's usage; `log_system_memory` adds
  whole-node memory; `log_per_gpu` breaks GPU stats out per device.

### ParameterCounter

Logs a model-size and frozen/trainable parameter summary once at startup — the
fastest way to confirm a freeze/LoRA config froze what you intended.

`ParameterCounter(show_details=True, show_gradients=True, sort_by_params=True, min_params=0)`

- Runs at startup; `show_gradients` separates trainable from frozen params,
  `sort_by_params` orders the breakdown, `min_params` hides tiny modules.

:::{seealso}
There is also a `LogitEntropyLogger` in the same module for classification-head
entropy. It follows the same `log_frequency` / `prefix` pattern.
:::

## Mutating hooks

These change runtime state. Order them carefully relative to the optimizer and
each other.

### WeightDecayExclusion

Rewrites the optimizer's parameter groups **before training** so that biases,
norm/gamma parameters, learnable tokens, and any 1-D parameter are excluded from
weight decay — the standard transformer recipe.

`WeightDecayExclusion(exclude_bias_from_wd=True, exclude_norm_from_wd=True, exclude_gamma_from_wd=True, exclude_token_from_wd=True, exclude_ndim_1_from_wd=True)`

- Splits params into decay / no-decay groups and marks the no-decay groups so a
  scheduler can skip them.

### WeightDecayScheduler

Updates weight decay **before each step** along a cosine schedule (common for
DINO/Sonata-style SSL, where WD ramps up over training).

`WeightDecayScheduler(base_value=0.04, final_value=0.2, warmup_ratio=1.0)`

- Builds a `CosineScheduler` in `before_train()` and seeds its iteration from
  `trainer.global_step` so the schedule is **resume-aware**.
- In `before_step()` it sets `weight_decay` only on param groups marked
  `apply_wd=True` (groups marked `apply_wd=False`, e.g. from
  {py:class}`~pimm.engines.hooks.optimizer.WeightDecayExclusion`, keep `0.0`), and logs `params/wd` to the writer.

:::{important}
Pair these two: put `WeightDecayExclusion` **before** `WeightDecayScheduler` in
`cfg.hooks` so the scheduler only touches the decay groups the exclusion hook
defined. The scheduler's total length comes from `cfg.scheduler.total_steps`.
:::

### DtypeOverrider

Forces selected modules (and optionally their parameters) to a specific compute
dtype — e.g. keeping a numerically sensitive decoder in `float32` while the rest
of the model runs in lower precision.

`DtypeOverrider(patterns=None, class_patterns=None, dtype="float32", methods_to_override=None, override_parameters=False, verbose=False, check_interval=10)`

- `patterns` match module **names** (regex); `class_patterns` match module
  **class names**.
- Wraps `methods_to_override` (default `["forward"]`) to cast in/out; with
  `override_parameters=True` it also casts the matched parameters.
- `check_interval` re-asserts the override periodically.

:::{tip}
For LArTPC detectors the panoptic/instance decoder typically must stay fp32.
`DtypeOverrider` is the surgical way to pin just that submodule without forcing
the whole model to full precision.
:::

### AttentionMaskAnnealingHook

Drives a model's attention-mask annealing schedule, for models that gradually
relax a structured attention mask over training.

`AttentionMaskAnnealingHook(log_frequency=100, log_per_layer=False, prefix="anneal")`

- Calls `model.update_anneal_step()` and logs the annealing factor(s) under
  `prefix`, **only when the model exposes that interface** — otherwise it is a
  no-op.

### GarbageHandler

Operational memory hook in `resources.py`.

`GarbageHandler(interval=150, disable_auto=True, empty_cache=False)`

- `disable_auto=True` turns off Python's automatic GC and collects manually
  every `interval` steps (predictable pauses instead of random GC stalls).
- `empty_cache=True` also empties the CUDA cache on that cadence.

### RuntimeProfiler

Wraps a few steps in `torch.profiler` to produce a trace and a top-ops table —
for one-off performance investigations, not steady-state runs.

`RuntimeProfiler(forward=True, backward=True, interrupt=False, warm_up=2, sort_by="cuda_time_total", row_limit=30, memory=True)`

- Profiles forward and/or backward after `warm_up` steps; writes traces under
  `cfg.save_path/logdir/`; `memory=True` records a CUDA memory history.
- `interrupt=True` stops the run after profiling — use it to grab a trace
  without paying for a full job.

:::{warning}
`RuntimeProfiler` adds real overhead and writes large trace files. Enable it for
a short diagnostic run and remove it from `cfg.hooks` for production training.
:::

## Quick reference

```{list-table}
:header-rows: 1
:widths: 30 18 52

* - Hook
  - Kind
  - Use it when…
* - `GradientNormLogger`
  - monitor
  - loss diverges/plateaus; suspect exploding or dead gradients
* - `PrototypeUsageLogger`
  - monitor
  - prototype SSL — watch for assignment collapse
* - `FeatureStdMonitor`
  - monitor
  - self-distillation — watch for feature-std collapse
* - `ResourceUtilizationLogger`
  - monitor
  - right-sizing jobs, hunting CPU/GPU memory leaks
* - `ParameterCounter`
  - monitor
  - confirming a freeze / LoRA / warm-start touched the right params
* - `WeightDecayExclusion`
  - mutator
  - excluding bias/norm/1-D params from weight decay
* - `WeightDecayScheduler`
  - mutator
  - cosine weight-decay ramp (DINO/Sonata SSL)
* - `DtypeOverrider`
  - mutator
  - pinning a sensitive submodule to fp32
* - `AttentionMaskAnnealingHook`
  - mutator
  - models with an attention-mask anneal schedule
* - `GarbageHandler`
  - mutator
  - controlling GC / CUDA-cache pauses
* - `RuntimeProfiler`
  - mutator
  - one-off performance / memory profiling
```

## Next

- {doc}`logging` — the everyday console / scalar / TensorBoard hooks.
- {doc}`writing_hooks` — build your own monitor with forward hooks and a clean
  `after_train()` teardown.
- {doc}`../checkpoints/hooks` — savers, and why evaluators come before them.
