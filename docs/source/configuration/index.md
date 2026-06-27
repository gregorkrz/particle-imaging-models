# Configuration

A pimm training config is an **executable Python file** under `configs/`. It
defines the model, dataset, transforms, optimizer, scheduler, trainer, hooks, and
runtime scalars as plain Python variables, and it is the source of truth for
*what* gets trained. (*How* and *where* a run executes — site, Slurm, container,
resources — lives in the launch YAML the launchers read; see
{doc}`../getting_started/quickstart`.)

:::{seealso}
For the one-paragraph mental model — "configs are Python, execution is YAML" —
read {doc}`../getting_started/concepts` first.
:::

## Anatomy of a config

Most configs inherit a runtime base, override a few scalars at the top, then
define `model`, `optimizer`/`scheduler`, `data`, and `hooks`:

```python
_base_ = ["../../_base_/default_runtime.py"]

# Runtime and logging
batch_size = 48
num_worker = 24
enable_amp = True
amp_dtype = "bfloat16"
seed = 0
use_wandb = True
wandb_project = "..."

# Shared constants (ordinary Python — reusable below)
grid_size = 0.001
warmup_ratio = 0.05

# Model
model = dict(type="...", ...)

# Optimizer and scheduler
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=base_wd)
scheduler = dict(type="OneCycleLR", ...)
param_dicts = [...]

# Data
transform = [...]
test_transform = [...]
data = dict(
    num_classes=...,
    names=[...],
    train=dict(type="...", transform=transform, ...),
    val=dict(type="...", transform=test_transform, ...),
    test=dict(type="...", transform=test_transform, ...),
)

# Hooks and tester
hooks = [...]
test = dict(type="...", ...)
```

A few conventions hold across every config:

- `model.type`, the dataset `type`, each hook `type`, `optimizer.type`,
  `scheduler.type`, `train.type` (trainer), and `test.type` (tester) are all
  **registry names**. See {doc}`registered models <../api/registry/models>` for model `type` strings.
- `data.train` / `data.val` / `data.test` must be complete enough for the dataset
  builder; inherited nested keys remain unless replaced.
- Transform lists are ordered pipelines (see {doc}`../datasets/index`). A child
  that assigns a transform list **replaces** the inherited list.
- W&B settings are normal config keys consumed by hooks;
  {py:class}`~pimm.engines.hooks.logging.WandbNamer` can derive run names from
  config paths like `model.type` or `data.train.max_len`.

### Worked example: layer-wise learning rates

Because a config is just Python, you can *compute* entries. `param_dicts` (the
optimizer's per-parameter-group settings) is a plain list, so this real example
(from `configs/panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft.py`) decays each
encoder block's LR toward the earliest, finest stages, keeps the random head on
the fast head LR, and threads the resulting LRs into the scheduler:

```python
optimizer = dict(type="AdamW", lr=head_lr, weight_decay=base_wd)

enc_depths = model["backbone"]["enc_depths"]
param_dicts = []

for e in range(len(enc_depths)):
    for b in range(enc_depths[e]):
        exp = sum(enc_depths) - sum(enc_depths[:e]) - b - 1
        param_dicts.append(
            dict(keyword=f"enc{e}.block{b}.", lr=encoder_lr * (lr_decay**exp))
        )

# Non-block backbone params (embedding/downsample/norm) fine-tune with the encoder.
param_dicts.append(dict(keyword="backbone.", lr=encoder_lr))
del enc_depths

scheduler = dict(
    type="OneCycleLR",
    max_lr=[head_lr] + [g["lr"] for g in param_dicts],
    pct_start=0.05,
    anneal_strategy="cos",
)
```

Note the `del enc_depths`: a local name that should not become a config key can
be deleted before the module finishes loading (see [mechanics](#config-file-mechanics)).

## Inheritance with `_base_`

Configs inherit from one or more base files with `_base_`, resolved relative to
the child file:

```python
_base_ = ["../../_base_/default_runtime.py"]
```

```python
_base_ = [
    "../pretrain/pretrain-sonata-v1m1-pilarnet-smallmask.py",
    "../other_base.py",
]
```

pimm loads all bases first, then recursively merges the child into the merged
base; child values win. Duplicate top-level keys **across multiple bases** are
rejected before the child merges.

### Dicts merge recursively; lists replace

For dictionary values, inheritance is recursive — the child patches only the keys
it names:

```python
# base
data = dict(train=dict(max_len=1_000_000, loop=1))
# child
data = dict(train=dict(max_len=1000))
# resolved
data = dict(train=dict(max_len=1000, loop=1))
```

Lists are **not** patched element-by-element. A child that assigns `hooks`,
`transform`, `criteria`, or any other list **replaces the whole list** — if you
only meant to change one entry, restate the full intended list.

For hook-only edits, prefer `hooks_override` when the target hook type already
exists (applied *before* CLI options; the helper key is removed from the resolved
config):

```python
hooks_override = {
    "WandbNamer": {"extra": "scratch"},
    "SemSegEvaluator": {"every_n_steps": 500},
}
```

### `_delete_=True` — replace instead of merge

To discard an inherited dict entirely and substitute a new one — when the
inherited value has the wrong shape or meaning — set `_delete_=True` in the child:

```python
model = dict(_delete_=True, type="panda-ar-v2m1", ar_hidden_dim=384)
scheduler = dict(_delete_=True, type="ExpLR", gamma=1.0)
```

### Template variables

Before execution, the loader substitutes file-local template strings:

| Template | Expands to |
| --- | --- |
| `{{ fileDirname }}` | directory of the config file |
| `{{ fileBasename }}` | file name with extension |
| `{{ fileBasenameNoExtension }}` | file name without extension |
| `{{ fileExtname }}` | file extension |

It also supports `{{ _base_.path.to.key }}`, replaced after the bases load — handy
when a child needs a nested base value without importing the base as Python.

## Runtime defaults

Most configs inherit `configs/_base_/default_runtime.py`, which defines the
baseline runtime contract:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Group
  - Keys
* - Checkpoint inputs
  - `weight`, `resume`
* - Run mode
  - `evaluate`, `test_only`
* - Reproducibility / output
  - `seed`, `save_path`
* - Loader shape
  - `num_worker`, `batch_size`, `batch_size_val`, `batch_size_test`
* - Schedule shape
  - `epoch`, `eval_epoch`
* - Numerics
  - `clip_grad`, `sync_bn`, `enable_amp`, `amp_dtype`, `matmul_precision`,
    `deterministic`
* - Distributed / training
  - `find_unused_parameters`, `prefetch_factor`, `detect_anomaly`, `mix_prob`,
    `param_dicts`
* - Defaults
  - `hooks`, `train` (trainer), `test` (tester)
```

The default hook list is a good reference for the lifecycle pieces every run gets:

```python
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="FinalEvaluator", test_last=False),
]
train = dict(type="DefaultTrainer")
test = dict(type="SemSegTester", verbose=True)
```

### Fine-tuning a backbone

Fine-tuning configs commonly remap checkpoint keys with a
{py:class}`~pimm.engines.hooks.checkpoint.CheckpointLoader` hook so a pretrained
encoder lands under the right submodule:

```python
hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.student.backbone",
        replacement="module.backbone",
    ),
    ...,
]
```

See {doc}`../checkpoints/saving_and_loading` for the remap mechanics this
consumes.

## Config file mechanics

Config files are regular Python modules. On load, pimm executes the file in a
temporary module and collects every global name that does **not** start with
`__` — which is why local variables, comprehensions, imports, arithmetic, and
values derived from earlier entries all work, and why a throwaway local can be
`del`-eted so it doesn't become a config key.

:::{important}
The loader is **Python-first**. `Config._file2dict` accepts `.py`, `.json`,
`.yaml`, and `.yml`, but JSON/YAML training-config parsing is *not* implemented —
it raises `NotImplementedError`. Use Python files for training configs. Launch
YAML is a separate system used only by the launchers.
:::

:::{warning}
Reserved top-level names are `filename`, `text`, and `pretty_text`. Don't use
them as config keys.
:::

### Registering custom components

A `type` is only resolvable if pimm has imported the class that registered it.
Register custom models/datasets/transforms/hooks by importing them from the
relevant package `__init__.py` (e.g. `pimm/datasets/__init__.py`) — not from the
config itself.

## CLI overrides

You rarely run `pimm/train.py` by hand — the launchers wrap it (pass bare
`KEY=VALUE` tokens after `--`; see {doc}`../getting_started/quickstart`). What's
config-specific is *how those overrides merge*. Each item is `KEY=VALUE`; dotted
keys create or update nested entries; values parse as `int` → `float` →
`true`/`false` → list/tuple → string:

```bash
--options epoch=10 enable_amp=true amp_dtype=bfloat16
--options data.train.max_len=1000 optimizer.lr=3e-5
--options model.backbone.order='[hilbert,hilbert-trans,z,z-trans]'
```

Unknown keys are **added**, not rejected — a typo may instead surface later
during model/dataset/hook/optimizer construction. CLI merges use the same
recursive dict merge as file inheritance, with two conveniences:

```bash
--options hooks.6.every_n_steps=500            # numeric segment patches a list element
--options hooks.SemSegEvaluator.every_n_steps=500   # address a hook by its type name
```

The `hooks.HookType.param` form is split out and applied after the generic merge,
so you don't depend on a fragile numeric index. `scripts/train.sh` always
provides `--options save_path=<experiment-dir>`; extra options are appended and
can override the same keys.

## Saved artifacts

For a fresh run (`resume` false), the parser creates `cfg.save_path/model` and
writes these under `cfg.save_path`:

```{list-table}
:header-rows: 1
:widths: 30 70

* - File
  - Contents
* - `config.py`
  - Resolved Python config after inheritance, `hooks_override`, CLI options, and
    seed init. Comments and `_base_` structure are not preserved.
* - `resolved_config.json`
  - JSON form of the full resolved config.
* - `model_config.json`
  - JSON form of `cfg.model`, when a `model` section exists.
* - `run_metadata.json`
  - command, cwd, host, original + absolute config paths, CLI options, save path,
    resume flag, and tracked-file git metadata.
```

These are **not** rewritten on resume. Treat the saved `config.py` and
`run_metadata.json` as the authoritative record of what a run started with — and
remember resume loads the saved `config.py`, not the original under `configs/`.

## Experiment variants

Use the smallest mechanism that still leaves a clear record:

```{list-table}
:header-rows: 1
:widths: 34 66

* - Use
  - For
* - CLI `--options` (post-`--` tokens)
  - Quick checks, temporary limits, LR probes, launcher-owned overrides.
* - A child Python config
  - Variants that should be reusable, reviewable, resumed, or compared in reports.
* - `launch/runs/*.yaml`
  - When the launch itself has state: special resources, checkpoint weights,
    resume behavior, chained jobs, W&B group names.
```

The recommended pattern — inherit a baseline and override only what changes:

```python
_base_ = ["../pretrain/pretrain-sonata-v1m1-pilarnet-smallmask.py"]

epoch = 100
data = dict(train=dict(max_len=100000), val=dict(max_len=1000))
model = dict(momentum_base=0.9995, momentum_final=0.9995)
optimizer = dict(lr=3e-5, weight_decay=0.2)
hooks_override = {"WandbNamer": {"extra": "tail-wd20"}}
```

:::{warning}
Avoid these anti-patterns:

- Editing a baseline config in place for a new experiment.
- Encoding Slurm accounts, partitions, container images, or site paths in Python
  configs.
- Relying on a long CLI override as the only record of a meaningful model or data
  change.
- Partially replacing an inherited list without restating the full intended list.
:::

Before an important run, dry-run it and read the rendered command — the best
source for the final config path, run name, resources, account, and overrides:

```bash
pimm launch --dry-run --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

## See also

- {doc}`../getting_started/quickstart` — `pimm launch` / `submit` / `export` and how to pass overrides.
- {doc}`registered models <../api/registry/models>` — model `type` strings to drop into `model.type`.
- {doc}`../getting_started/concepts` — registries, the packed batch, the trainer contract.
- {doc}`../hooks/index` — the hooks you wire up in the `hooks` list.
- {doc}`../checkpoints/index` — what fine-tuning and resume consume.
