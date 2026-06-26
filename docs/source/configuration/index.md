# Configuration

pimm training configs are **executable Python files** under `configs/`. They
define the model, dataset, optimizer, scheduler, trainer, hooks, and runtime
settings as plain Python variables. The launchers and shell scripts wrap these
files, but the Python config is always the source of truth for *what* gets
trained.

:::{seealso}
This page is the deep dive. For the one-paragraph mental model — "configs are
Python, execution is YAML" — read {doc}`../getting_started/concepts` first. For
*how* to run a config, see {doc}`../reference/cli`.
:::

There are three related layers, and they own different things:

```{list-table}
:header-rows: 1
:widths: 28 72

* - Layer
  - Owns
* - **Python configs** (`configs/*.py`)
  - *What* to train: model, dataset, transforms, optimizer, scheduler, hooks,
    epochs, batch size. Loaded by {py:class}`~pimm.utils.config.Config`.
* - **`scripts/train.sh` / `test.sh`**
  - Normalize config paths, choose experiment directories, snapshot code, and
    pass runtime overrides.
* - **Launch YAML** (`launch/`)
  - *How / where* to run: site, Slurm directives, container, resources, run
    naming, resume, checkpoint weights, env vars. Composed by `pimm launch`
    / `pimm submit`.
```

Keep scheduler, account, container, and site-path choices in launch YAML. Keep
model and dataset behavior in Python configs.

## Python config files

Config files are regular Python modules. On load, pimm executes the file in a
temporary module and collects every global name that does **not** start with
`__`. This means a config can use local variables, comprehensions, imports,
arithmetic, and values derived from earlier entries:

```python
grid_size = 0.001
warmup_ratio = 0.05

model = dict(
    type="Sonata-v1m1",
    mask_jitter=grid_size / 2,
)

param_dicts = [
    dict(keyword=f"enc{e}.block{b}.", lr=base_lr * lr_decay**b)
    for e in range(5)
    for b in range(3)
]
```

:::{important}
The loader is **Python-first**. `Config._file2dict` accepts `.py`, `.json`,
`.yaml`, and `.yml` suffixes, but JSON/YAML training-config parsing is *not*
implemented on this path — it raises `NotImplementedError`. Use Python files for
training configs. Launch YAML is a separate system used only by the launchers.
:::

:::{warning}
Reserved top-level names are `filename`, `text`, and `pretty_text`. Do not use
them as config keys.
:::

### Custom imports

Most models, datasets, transforms, hooks, optimizers, and schedulers are found
through pimm registries by their `type` string. Registration happens by **import
side effect** — a class that is never imported is not buildable from config. If a
config needs a side-effect import to register a module, import it directly:

```python
__import__("pimm.datasets.lucid_event_ssl")
__import__("pimm.engines.hooks.lucid_event_probe")
```

The loader also supports the MMCV-style `custom_imports` key:

```python
custom_imports = dict(
    imports=["pimm.datasets.lucid_event_ssl"],
    allow_failed_imports=False,
)
```

:::{warning}
A config `__import__` is enough for a fresh run, but **resume reloads the dumped
`config.py`**, which has no `__import__` line. Custom datasets/hooks that must
survive a resume should be registered in the package `__init__`, not only in the
config. See {doc}`../getting_started/concepts`.
:::

## Inheritance with `_base_`

Configs inherit from one or more base files with `_base_`. Paths are resolved
relative to the child config file:

```python
_base_ = ["../../_base_/default_runtime.py"]
```

or several bases:

```python
_base_ = [
    "../pretrain/pretrain-sonata-v1m1-pilarnet-smallmask.py",
    "../other_base.py",
]
```

pimm loads all bases first, then recursively merges the child into the merged
base. Child values override base values. Duplicate top-level keys across multiple
base files are rejected before the child is merged.

### Recursive dict merge

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

### `_delete_=True` — replace instead of merge

To discard the inherited dictionary entirely and substitute a new one, set
`_delete_=True` inside the child dict:

```python
model = dict(
    _delete_=True,
    type="panda-ar-v2m1",
    ar_hidden_dim=384,
)
```

```python
scheduler = dict(_delete_=True, type="ExpLR", gamma=1.0)
```

Use `_delete_=True` when the inherited value has the wrong shape or meaning — for
example replacing a detector model with an AR policy, or swapping a `OneCycleLR`
scheduler for `ExpLR`.

### Lists replace, they do not merge

Lists are **not** patched element-by-element during file inheritance. A child
that assigns `hooks`, `transform`, `criteria`, or any other list **replaces the
whole list**. If you only meant to change one entry, you must restate the full
intended list.

For hook-only edits, prefer `hooks_override` when the target hook type already
exists:

```python
hooks_override = {
    "WandbNamer": {"extra": "scratch"},
    "SemSegEvaluator": {"every_n_steps": 500},
}
```

`default_config_parser` applies `hooks_override` *before* CLI options and removes
the helper key from the resolved config.

### Template variables

Before execution, the loader substitutes file-local template strings:

| Template | Expands to |
| --- | --- |
| `{{ fileDirname }}` | directory of the config file |
| `{{ fileBasename }}` | file name with extension |
| `{{ fileBasenameNoExtension }}` | file name without extension |
| `{{ fileExtname }}` | file extension |

It also supports references to base-config values with `{{ _base_.path.to.key }}`,
replaced after the base configs are loaded. This is useful when a child needs to
copy a nested base value without importing the base file as Python.

## Runtime defaults

Most training configs inherit from `configs/_base_/default_runtime.py`, which
defines the baseline runtime contract:

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

The default hook list is a good reference for the lifecycle pieces every run
gets:

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

Experiment configs usually override most of these runtime scalars near the top of
the file, then define `model`, `optimizer`, `scheduler`, `data`, and `hooks`.

## Anatomy of a config

Most pimm configs follow this shape:

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

# Shared constants
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

# Hooks and tester overrides
hooks = [...]
test = dict(type="...", ...)
```

Section conventions:

- `model.type`, dataset `type`, hook `type`, optimizer `type`, scheduler `type`,
  trainer `train.type`, and tester `test.type` are **registry names**. See
  {doc}`../reference/model_zoo` for model `type` strings.
- `data.train`, `data.val`, and `data.test` must be complete enough for the
  dataset builder; inherited nested keys remain unless replaced.
- Transform lists are ordered pipelines. A child that assigns a transform list
  replaces the inherited list. See {doc}`../datasets/index`.
- `param_dicts` is consumed by optimizer construction for parameter-specific
  settings such as layer-wise learning rates (see below).
- W&B settings are normal config keys consumed by hooks. {py:class}`~pimm.engines.hooks.logging.WandbNamer` can derive
  run names from config paths like `model.type` or `data.train.max_len`.

### Worked example: layer-wise learning rates

`param_dicts` is just a list, so you can build it with ordinary Python. This
real example (from
`configs/panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft.py`) decays
each encoder block's LR toward the earliest, finest stages, leaves the random
head on the fast head LR, and threads the resulting LRs into the scheduler:

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
simply be deleted before the module finishes loading.

### Fine-tuning a backbone

Fine-tuning configs commonly remap checkpoint keys with a {py:class}`~pimm.engines.hooks.checkpoint.CheckpointLoader` hook
so a pretrained encoder lands under the right submodule:

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

See {doc}`../checkpoints/index` for the checkpoint formats this consumes.

## CLI `--options`

Training entry points parse CLI overrides with `DictAction`:

```bash
python pimm/train.py \
  --config-file configs/panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask.py \
  --options epoch=10 data.train.max_len=1000 model.backbone.drop_path=0.1
```

Each item must be `KEY=VALUE`. Dotted keys create or update nested entries. Values
are parsed in this order:

1. `int`, then `float`, then `true`/`false` booleans
2. comma-separated lists, bracketed lists, parenthesized tuples
3. otherwise a string

```bash
--options epoch=10 enable_amp=true amp_dtype=bfloat16
--options data.train.max_len=1000 optimizer.lr=3e-5
--options model.backbone.order='[hilbert,hilbert-trans,z,z-trans]'
```

Unknown keys are **added**, not rejected — a typo may instead fail later during
model, dataset, hook, optimizer, or scheduler construction.

CLI merges use the same recursive dict merge as file inheritance, with one extra
behavior: a **numeric** dotted segment can patch a list element that already
exists:

```bash
--options hooks.6.every_n_steps=500
```

`default_config_parser` also has a hook-type override helper for keys shaped like
`hooks.HookType.param=value`. It splits these out, runs the generic merge on the
rest, then applies the hook-type keys afterward — so you can address a hook by its
**type name** instead of a fragile numeric index:

```bash
--options hooks.SemSegEvaluator.every_n_steps=500
```

`scripts/train.sh` always provides `--options save_path=<experiment-dir>`; extra
options after `--` are appended and can override the same keys.

:::{note}
With the launchers, training overrides are bare `KEY=VALUE` tokens after `--`
(e.g. `pimm launch ... -- epoch=10`). The `-- --options ...` form is only for
direct `scripts/train.sh` invocations. See {doc}`../reference/cli`.
:::

## Saved artifacts

For a fresh run (where `resume` is false), `default_config_parser` creates
`cfg.save_path/model` and writes these files under `cfg.save_path`:

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

These files are **not** rewritten on resume. Treat the saved `config.py` and
`run_metadata.json` as the authoritative record of what a run started with — and
remember that resume loads the saved `config.py`, not the original file under
`configs/`.

## Experiment variants

Use the smallest mechanism that still leaves a clear record:

```{list-table}
:header-rows: 1
:widths: 34 66

* - Use
  - For
* - CLI `--options`
  - Quick checks, quick limits, temporary LR probes, launcher-owned overrides.
* - A child Python config
  - Variants that should be reusable, reviewable, resumed, or compared in reports.
* - `launch/runs/*.yaml`
  - When the launch itself has state: special resources, checkpoint weights,
    resume behavior, chained jobs, W&B group names.
```

Recommended variant pattern — inherit a baseline and override only what changes:

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

Before launching an important run, dry-run it and read the rendered command — it
is the best source for the final config path, run name, resources, account, and
training overrides:

```bash
pimm launch --dry-run --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```

## See also

- {doc}`../reference/cli` — `pimm launch` / `submit` / `export` and the direct
  scripts.
- {doc}`../reference/model_zoo` — model `type` strings to drop into `model.type`.
- {doc}`../getting_started/concepts` — registries, the packed batch, the trainer
  contract.
- {doc}`../hooks/index` — the hooks you wire up in the `hooks` list.
- {doc}`../checkpoints/index` — what fine-tuning and resume consume.
