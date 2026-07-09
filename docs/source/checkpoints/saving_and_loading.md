# Saving & loading

During training, checkpointing is driven by **hooks** in `cfg.hooks`: one loads weights (and, on resume, the full training state) before the loop, and one saves on a schedule.
Outside of training, you can load weights **programmatically** with `pimm.export`.
For what a checkpoint actually contains and the on-disk layout, see {doc}`index`.
To continue an interrupted run, see {doc}`resuming`.

## Saving

Three hooks (registered in `pimm/engines/hooks/checkpoint.py`) cover saving and
loading:

- {py:class}`~pimm.engines.hooks.checkpoint.CheckpointLoader` - in `before_train()`,
  loads weights from the config's `weight` attribute or restores the full training
  state from the config's `resume` attribute. Supports key remapping for fine-tuning.
- {py:class}`~pimm.engines.hooks.checkpoint.CheckpointSaver` - evaluator-aware
  saver. Writes the rolling checkpoint on a save step or eval step and in
  `after_train()`, and writes `model_best.pth` when the metric improves.
- {py:class}`~pimm.engines.hooks.checkpoint.CheckpointSaverIteration` - long-run
  saver on a pure global-step schedule; writes every `save_freq` steps and in
  `after_train()`.

:::{important}
**Hook order matters.** A saver can only mark a *best* checkpoint after an
evaluator has published `current_metric_value`. Put evaluator hooks **before**
the saver in the list.
Placing `InformationWriter` and the evaluator before the saver also ensures their scalars are published before the checkpoint is written, so those scalars are captured in the saved state.
:::

### Save frequency

::::{tab-set}

:::{tab-item} Epoch / eval-step frequency
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
{py:class}`~pimm.engines.hooks.checkpoint.CheckpointSaver` writes the rolling
checkpoint and, when the evaluator metric improves, `model_best.pth`.
`save_freq=None` means "save at eval points / end of training"; set an integer
for periodic `iter_<step>.pth` snapshots.
:::

:::{tab-item} Iteration frequency (long runs)
```python
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaverIteration", save_freq=1000),
]
```
{py:class}`~pimm.engines.hooks.checkpoint.CheckpointSaverIteration` writes a fresh
rolling checkpoint every `save_freq` optimizer steps and once more in
`after_train()`. This is the right saver for walltime-limited HPC runs and
requeue chains - pick a `save_freq` small enough that an attempt always leaves a
recent complete checkpoint before it times out.
:::

::::

:::{note}
Both savers build the payload with per-rank RNG and dataloader state, so they
**must be called on every rank** - a `standard`-format save is a collective DCP
operation. Rank-0-only side artifacts (`model_best.pth`, `iter_<step>.pth`) are
guarded internally. The deprecated `backend=` argument is an alias for
`checkpoint_format` (`dcp`→`standard`, `torch`→`legacy`).
The payload saves model, optimizer, scheduler, scaler, dataloader, RNG, and trainer state, but **not** arbitrary hook attributes.
:::

## Loading

There are two ways to load weights, depending on whether you're inside a training
run or not.
Outside a training run, build a ready-to-use model from an export or Hub repo with {py:func}`~pimm.from_pretrained`.
Loading from every source, the config precedence / drift-tolerance behavior, and the full inference walkthrough live on the canonical loading page - see {doc}`../research_ecosystem/using_trained_models`.

### In a training run: the `CheckpointLoader` hook

`CheckpointLoader.__init__(keywords="", replacement=None, replacements=None,
strict=False)`. Its behavior is set by the config:

- **Warm-start** (`cfg.weight=<path>`, `cfg.resume=False`) - loads *model weights
  only*. Optimizer, scheduler, step counter, and dataloader start fresh. This is
  the fine-tuning path.
- **Resume** (`cfg.resume=True`) - restores the *full* training state for exact
  continuation. See {doc}`resuming`.

### Fine-tune key remapping

A pretraining checkpoint's keys rarely line up one-to-one with a fine-tune
model. The classic case: a Sonata SSL run wraps a *student* and *teacher* copy of
the backbone, so its weights are keyed `student.backbone.*` / `teacher.backbone.*`,
but the fine-tune model needs only `backbone.*`. Remap with the hook:

```python
dict(
    type="CheckpointLoader",
    keywords="module.student.backbone",   # match keys starting with this…
    replacement="module.backbone",        # …rewrite the prefix to this
)
```

Keys are normalized in this order: strip a leading `module.`, apply your
`keywords → replacement` rewrite (only where the bare key *starts with*
`keywords`), then re-add `module.` when `world_size > 1`. Use `replacements={...}`
for several independent rewrites in a single load (so missing/unexpected keys are
reported truthfully).

:::{important}
A remap that matches **zero** parameters raises rather than silently training
from random init. If you see `No weight found` / `Missing keys: [...everything]`,
your `keywords` prefix is wrong. Judge a successful load by the loss curves of the
*new* head, not just the absence of errors.
:::

### Low-level state-dict helpers

For surgical loads - pulling one submodule into an already-built model, or
remapping keys by hand - the helpers in `pimm.export` are more direct than a full
`from_pretrained`:

```python
from pimm.models.builder import build_model
from pimm.export import load_pretrained

model = build_model(cfg.model)
load_pretrained(
    model.backbone,
    "exp/pretrain/model/model_last.pth",
    prefix="student.backbone.",   # keep keys with this prefix, then strip it
    remove_prefix=True,
    strict=False,
)
```

| Helper | Purpose |
|--------|---------|
| {py:func}`~pimm.export.load_pretrained` | load into an existing model with filtering/remapping |
| {py:func}`~pimm.export.load_state_dict_from_checkpoint` | load `.safetensors` or torch checkpoints → state dict |
| {py:func}`~pimm.export.clean_state_dict` | strip `module.` / `_orig_mod.` prefixes |
| {py:func}`~pimm.export.filter_state_dict_by_prefix` | keep keys with a prefix (optionally stripping it) |
| {py:func}`~pimm.export.remap_state_dict_keys` | exact-then-prefix key remapping |
