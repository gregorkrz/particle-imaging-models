# Evaluation

Evaluation in pimm is just more hooks. An **evaluator hook** runs validation
inside the training loop, computes task metrics, and writes a single
*selection metric* into `comm_info` so a checkpoint saver can mark
`model_best.pth`. At the end of training, a **final tester** runs the held-out
test set. For SSL pre-training, **probe evaluators** fit lightweight linear
probes on frozen features to track representation quality without a task head.

The evaluator hooks live under `pimm/engines/hooks/eval/` (task evaluators at
the top level; pre-training probes under `eval/pretrain/`).

:::{seealso}
Evaluators are hooks — {doc}`../hooks/index` covers the lifecycle and the
`comm_info` / `storage` / `writer` channels used throughout this page.
:::

## The selection-metric contract

Every evaluator that can drive checkpoint selection writes two keys:

```python
trainer.comm_info["current_metric_value"] = mIoU      # float, higher = better
trainer.comm_info["current_metric_name"]  = "mIoU"
```

A {py:class}`~pimm.engines.hooks.checkpoint.CheckpointSaver` reads these *after* the evaluator runs and writes
`model_best.pth` only when the value improves on `best_metric_value`.

:::{important}
**Order: evaluator before saver.** If the saver runs first it sees a stale (or
missing) metric and never marks a best checkpoint. This is the single most
common evaluation-config mistake — see {doc}`../checkpoints/hooks`.
:::

```text
after_step()/after_epoch():
   Evaluator ──writes──▶ comm_info["current_metric_value"/"current_metric_name"]
        │
   CheckpointSaver ──reads──▶ improved? → model_best.pth
```

## Trigger modes

Every evaluator takes `every_n_steps`, which selects when it fires:

```{list-table}
:header-rows: 1
:widths: 28 72

* - `every_n_steps`
  - Behavior
* - `> 0`
  - Runs from `after_step()` when the global step is divisible by the interval —
    use this for long single-epoch / iteration-based runs.
* - `0` (default)
  - Runs from `after_epoch()` — one validation pass per epoch.
```

All evaluators are gated by `cfg.evaluate`: with `evaluate=False` the validation
and test loaders aren't even built, and the evaluators no-op. Most use
`trainer.val_loader`.

## Task evaluators

```{list-table}
:header-rows: 1
:widths: 26 50 24

* - Hook
  - Computes
  - Selection metric
* - `SemSegEvaluator`
  - loss, per-class intersection/union/target, mIoU, mAcc, allAcc, macro
    precision/recall/F1; optional per-class and per-instance majority-vote
    metrics.
  - **mIoU**
* - `InstanceSegmentationEvaluator`
  - instance-level ARI, detection/class stats, optional momentum-regression
    metrics, multi-label outputs.
  - primary-label metric
* - `PretrainEvaluator`
  - frozen-feature linear-probe grid over one or more label keys.
  - **mF1**
* - `MAEEvaluator` / `HMAEEvaluator`
  - reconstruction loss (avg / coord / feat) and mask ratio.
  - **negative val loss**
```

A few specifics worth knowing:

- **{py:class}`~pimm.engines.hooks.eval.semantic_segmentation.SemSegEvaluator`** is the standard segmentation validator; `write_cls_iou`
  adds per-class IoU to the log. Its selection metric is **mIoU**.
- **{py:class}`~pimm.engines.hooks.eval.instance_segmentation.InstanceSegmentationEvaluator`** requires **validation batch size 1**
  (instance/ARI bookkeeping is per-event). It can emit momentum-regression
  metrics and multiple label heads, setting the current metric for the primary
  label. The `InsegTrainer` provides the matching instance collation.
- **`PretrainEvaluator`** extracts frozen point features via a backbone,
  `encode()`, or the `return_point=True` path, splits validation events into a
  small probe train/test set, and trains a grid of linear classifiers
  (`eval/pretrain/linear.py`). It stores **mF1**.
- **{py:class}`~pimm.engines.hooks.eval.pretrain.mae.MAEEvaluator` / `HMAEEvaluator`** validate reconstruction and use
  **negative validation loss** as the metric, because lower loss is better and
  the saver always treats *higher = better*.

:::{seealso}
What an evaluator reads from the model output dict (`seg_logits`, `point`,
`pred_masks`, …) is summarized in the output-key table in
{doc}`../getting_started/concepts`.
:::

## SSL probe evaluators (LUCiD / pre-training)

For self-supervised pre-training there is no task head to score, so probe
evaluators read frozen features and fit a small classifier. These are stricter
about data leakage than task evaluators.

### OnlineLinearProbe

A cheap, always-on probe that trains a detached classifier *during* training in
`after_step()`.

- Expects the model to expose `_probe_feat` and `_probe_segment_motif` on the
  unwrapped model.
- Lazily builds a `LayerNorm + Linear` head, trains it with its own AdamW, and
  logs accuracy, loss, and per-class / macro precision/recall/F1.
- The probe lives **on the hook, not on `trainer.model`** — so the checkpoint
  payload does **not** save the probe weights or its optimizer. A resumed run
  re-grows the probe from scratch (fine for a diagnostic signal).

### EventLinearProbeEvaluator

The LUCiD event-level probe — the trustworthy signal for water-Cherenkov SSL
quality. It is deliberately conservative:

- Runs on **rank 0** and synchronizes the other ranks around evaluation.
- **Requires heldout data.** It accepts split metadata `holdout`, `val`, or
  `test`, and **rejects** `train` or `all`.
- **Verifies train/heldout disjointness** when datasets expose `data_list` and
  `datasets`; nested `Subset` wrappers are respected.
- Calls `model(input_dict, return_point=True)`, **mean-pools point features per
  event**, splits heldout events into probe-train/probe-test by class, and
  trains a linear probe via `LinearProbingTrainer`.
- Logs mIoU, macro precision/recall/F1, per-class metrics, and a confusion
  matrix; uses **event-probe mF1** as the current metric.

:::{warning}
**Heldout means heldout.** A config shared between train and probe-eval must sit
at the *same list index* in both, because LUCiD seeds its holdout by list
position. A mismatch causes train/heldout leakage — and
{py:class}`~pimm.engines.hooks.eval.pretrain.lucid_event_probe.EventLinearProbeEvaluator` will raise rather than report an optimistic number.
`tests/test_lucid_event_probe.py` covers the heldout guard, leakage rejection,
and nested-subset event keys.
:::

### EventProbeSuiteEvaluator

Runs a **suite** of probes (multiple PID classification tasks plus momentum
regression) from a single shared feature-extraction pass, so you get a panel of
metrics for the cost of one forward sweep.

`EventProbeSuiteEvaluator(data, tasks, every_n_steps=0, train_fraction=0.5,
seed=0, prefix="probe_suite", batch_size=16, num_workers=0, ...,
require_heldout_data=True, write_cls_metrics=False)`

- `tasks` is a list of `{"type": "classification", "classes": [...]}` and
  `{"type": "regression", "configs": [...], "target": ...}` entries; each is
  validated at construction.
- Applies the **same heldout guard** as `EventLinearProbeEvaluator`
  (`require_heldout_data=True`).
- Builds its own dataset/loader from the `data` config and shares one feature
  pass across every task.

:::{tip}
Prefer the probe-suite (full holdout, one shared pass) over the per-step
`event_probe/val` numbers, which only see rank-0's `1/world_size` shard of the
val loader and read biased-low and jumpy.
:::

## Final testing

{py:class}`~pimm.engines.hooks.eval.final_tester.FinalEvaluator` runs once in `after_train()` and is the in-process equivalent of
the standalone tester.

- Skips entirely when `cfg.evaluate=False`.
- Builds a tester from `cfg.test.type` (e.g.
  `cfg.test = dict(type="SemSegTester", ...)`).
- By default **loads `model_best.pth`** before testing; `test_last=True`
  evaluates the *current* model state instead.

```python
cfg.test = dict(type="SemSegTester")
hooks = [
    # ... evaluators, then saver ...
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="FinalEvaluator", test_last=False),   # tests model_best by default
]
```

### Testing a finished run from the shell

To re-test an existing experiment without retraining, use the direct tester. It
runs `pimm/test.py` against the experiment's **saved code snapshot**, so results
reflect the code the run was trained with, not your current working tree.

```bash
sh scripts/test.sh -c <config-path> -n <experiment-name> -w model_best
```

```{list-table}
:header-rows: 1
:widths: 22 78

* - Flag
  - Meaning
* - `-c CONFIG`
  - config path under `configs/`, with or without `.py`
* - `-n NAME`
  - experiment name (defaults to `debug`)
* - `-w WEIGHT`
  - checkpoint stem under `exp/.../model`; defaults to `model_best`
* - `-p PYTHON`
  - Python interpreter
```

Under the hood the script sets
`PYTHONPATH=./exp/<config-group>/<name>/code` and invokes:

```bash
--options save_path=<exp-dir> weight=<exp-dir>/model/<weight>.pth
```

:::{note}
`scripts/test.sh` parses `-g`/`-m` for symmetry with `train.sh` but does **not**
wrap testing in `torchrun`; it runs a single process. The tester class itself is
chosen by `cfg.test.type`.
:::

## Next

- {doc}`../checkpoints/hooks` — savers and the `model_best.pth` selection logic.
- {doc}`../hooks/writing_hooks` — write a custom evaluator that drives selection.
- {doc}`../datasets/index` — the held-out split conventions probes depend on.
