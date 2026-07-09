# Quickstart

This page will walk you through a full pre-train run and subsequent fine-tuning of Panda on the PILArNet-M dataset. It assumes you've {doc}`installed pimm <installation>`, have an NVIDIA GPU, and have
**PILArNet-M downloaded** (see {doc}`../datasets/pilarnet`).

## Configure environment

Scripts load a `.env` from the repo root if present, so copy the template and
edit it:

```bash
cp example.env .env && nano .env
```

See {doc}`installation` for the full environment-variable table.

Download PILArNet-M so training has data on disk:

```bash
uv run python scripts/download_pilarnet.py --version v2 --output-dir /path/to/dataset
```

Set `PILARNET_DATA_ROOT_V2` in `.env` to the downloaded directory.
See {doc}`../datasets/pilarnet` for the layout and revision details.

## Main command

The primary entry point for local GPU(s) is `pimm launch`, and on SLURM-managed clusters `pimm submit`:

```bash
uv run pimm launch --train.config <config-path> [-- KEY=VALUE ...]
uv run pimm submit --site <site> --train.config <config-path> [-- KEY=VALUE ...]
```

`--train.config` is a path under `configs/`, with or without `.py`. Everything
after a bare `--` separator is a training override, written as `KEY=VALUE`.

pimm has three console commands (all also reachable as `uv run python -m pimm.cli …`):

```{list-table}
:header-rows: 1
:widths: 22 78

* - Command
  - Purpose
* - `pimm launch`
  - Run training locally or inside an existing allocation (**local only**).
* - `pimm submit`
  - Submit training to Slurm (see {doc}`../hpc/index`).
* - `pimm export`
  - Export a checkpoint to a portable pretrained artifact.
```

## A real run

```bash
# single GPU
uv run pimm launch \
  --train.config panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft \
  --run.name semseg-pt-v3m2 \
  --resources.nproc-per-node 1

# four GPUs on one node (global batch_size splits to 4/GPU)
uv run pimm launch \
  --train.config panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft \
  --resources.nproc-per-node 4
```

See {doc}`../distributed/index` for how batch size and workers split across GPUs.

## Override config values

Two ways, smallest mechanism first:

::::{tab-set}

:::{tab-item} CLI override (quick probes)
```bash
uv run pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask \
  -- epoch=50 data.train.max_len=500000 optimizer.lr=3e-5
```
Bare `KEY=VALUE` tokens after `--`. Dotted keys patch nested config; values
parse as int → float → bool → list/tuple → string. Great for short-lived probes.
:::

:::{tab-item} Child config (reusable variants)
```python
# configs/panda/pretrain/my_variant.py
_base_ = ["./pretrain-sonata-v1m1-pilarnet-smallmask.py"]

epoch = 100
data = dict(train=dict(max_len=100000), val=dict(max_len=1000))
optimizer = dict(lr=3e-5, weight_decay=0.2)
```
Use a child config for anything you want to review, resume, or compare. See
{doc}`../configuration/index`.
:::

::::

:::{warning}
Training overrides after `--` must be bare `KEY=VALUE` tokens. Any token starting
with `--` is rejected: `-- --options epoch=10` is **invalid** for `pimm launch` /
`pimm submit` - write `-- epoch=10`.
:::

## What a run produces

By default the launcher snapshots the codebase and trains from that copy, so a
run is reproducible by construction:

```text
exp/<dataset>/<name>/
  code/                 # snapshot of pimm, scripts, tools (run trains from here)
  config.py             # resolved config after inheritance + overrides + seed
  resolved_config.json  # full resolved config as JSON
  model_config.json     # cfg.model as JSON
  run_metadata.json     # command, host, git status, original config path
  train.log             # training log
  model/                # checkpoints (see Checkpoints section)
```

Treat `config.py` and `run_metadata.json` as the authoritative record of what the
run started with. Set `MODEL_DIR` to put the (large) `model/` directory on
another disk; the experiment `model/` becomes a symlink to it.

## Resuming a run

If a run stops, you can continue it exactly (with RNG, dataloader position, step, and
optimizer state all restored, even mid-epoch):

```bash
uv run pimm launch \
  --train.config panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft \
  --run.name semseg-pt-v3m2 \
  --train.resume                # <-- new!
```

The experiment directory must already exist; the launcher selects the newest
complete checkpoint (`last`, `last.prev`, `model_last.pth`). See
{doc}`../checkpoints/resuming`.

## Logging

Rank 0 writes either Weights & Biases or TensorBoard:

```bash
uv run pimm launch --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask \
  --run.name test \
  --run.wandb-name test-display \
  --run.wandb-project Pretraining-Sonata-PILArNet-M
```

Set `use_wandb=False` to write TensorBoard events under the experiment directory
instead. See {doc}`../hooks/logging`.

## Evaluate a trained model

Evaluate the run's **snapshot** code on the held-out split:

```bash
uv run sh scripts/test.sh -c panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft \
  -n semseg-pt-v3m2 -w model_best
```

## Export and load a trained model

Turn a checkpoint into a portable artifact, then load it in Python:

```bash
uv run pimm export --run-dir exp/panda/semseg/semseg-pt-v3m2 last ./artifacts/my-model
```

```python
import pimm

model = pimm.from_pretrained("artifacts/my-model")   # ready for inference
print(type(model).__name__)                            # -> DefaultSegmentorV2
```

See {doc}`../evaluation/index`, {doc}`../checkpoints/exporting`, and
{doc}`../research_ecosystem/using_trained_models`.

## Scale by submitting to a cluster

The same run becomes a managed Slurm job by swapping `pimm launch` for `pimm
submit --site <site>`:

```bash
uv run pimm submit --site mycluster \
  --train.config panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft \
  --resources.nnodes 1 --resources.nproc-per-node 4 --resources.time 02:00:00
```

A site profile is a YAML file under `launch/sites/` describing your cluster - see {doc}`../hpc/sites`.

See {doc}`../hpc/index` for sites, interactive vs batch, recipes, and requeue
chaining, and {doc}`../distributed/index` for how batch size and workers split
across GPUs and nodes.

## Command reference

:::{dropdown} How the launchers resolve settings
`pimm launch` / `pimm submit` load settings in this order (later wins):

1. `launch/defaults.yaml`
2. `launch/sites/<site>.yaml`
3. an optional run recipe passed with `--recipe PATH`
4. CLI overrides and post-`--` training overrides

CLI flags are nested and dotted (`--train.config`, `--resources.nproc-per-node`, `--run.name`, `--slurm.account`).
There are no flat `--config`, `--option`, or `--set` flags.
:::

:::{dropdown} Full flag table
```{list-table}
:header-rows: 1
:widths: 38 32 30

* - Setting
  - Flag
  - Notes
* - training config
  - `--train.config`
  - path under `configs/`, `.py` optional
* - checkpoint weight
  - `--train.weight`
  - passed through as `weight=`
* - resume
  - `--train.resume`
  - resume from latest complete checkpoint
* - skip code snapshot
  - `--train.no-code-copy`
  - run from live repo (dev mode)
* - GPUs per node
  - `--resources.nproc-per-node`
  - torchrun processes per node
* - nodes
  - `--resources.nnodes`
  -
* - walltime
  - `--resources.time`
  - `HH:MM:SS`
* - run name
  - `--run.name`
  - default: auto-generated `<config>-<timestamp>`
* - fixed run dir (no timestamp)
  - `--run.no-timestamp`
  - write to exactly `exp/.../<name>/`
* - W&B display name / project
  - `--run.wandb-name` / `--run.wandb-project`
  -
* - Slurm account
  - `--slurm.account`
  - submit only
* - Slurm partition / QOS
  - `--slurm.partition` / `--slurm.qos`
  - submit only
* - run recipe
  - `--recipe PATH`
  - `launch/runs/*.yaml`
* - requeue attempts
  - `--chain.jobs N`
  - submit only (submitit requeue)
* - interactive salloc
  - `--interactive`
  - submit only
* - write rendered output
  - `--output PATH`
  - the launch script / submitit manifest
* - dry run
  - `--dry-run`
  - render without executing
```
:::

:::{dropdown} Config-path normalization
`--train.config` is normalized relative to `configs/` and without `.py`. These
are equivalent:

```bash
--train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
--train.config configs/panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask.py
--train.config /abs/path/configs/panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask.py
```
:::

:::{dropdown} `pimm export` flags
```bash
uv run pimm export --run-dir exp/panda/semseg/my_run last ./artifacts/my-model
```

The two positional arguments are the checkpoint and the output directory. With
`--run-dir`, the checkpoint name (`last`, `model_best`, …) resolves under
`<run-dir>/model`, and the config is inferred from `<run-dir>/config.py` /
`resolved_config.json`. Split checkpoint dirs (`model/last`) and legacy `.pth`
files both work.

```{list-table}
:header-rows: 1
:widths: 32 68

* - Flag
  - Meaning
* - `--run-dir PATH`
  - experiment directory with `config.py` and `model/`; lets the checkpoint arg
    be a name like `last`
* - `--config PATH`
  - config to use when it cannot be inferred from the run dir
* - `--model-card PATH`
  - a `README.md` text file to include as the model card
* - `--push-to-hub REPO_ID`
  - upload to the Hugging Face Hub after export
* - `--public`
  - create a public Hub repo (default: private)
* - `--token TOKEN`
  - Hugging Face token for upload
* - `--device DEVICE`
  - device used while consolidating tensors (default `cpu`)
* - `--no-safe-serialization`
  - write `model.bin` (torch pickle) instead of `model.safetensors`
* - `--dry-run`
  - print resolved paths and exit without writing
```

See {doc}`../checkpoints/index` for the artifact layout and the Hugging Face
workflow.
:::

:::{dropdown} Direct scripts - `scripts/train.sh` / `scripts/test.sh`
The launchers ultimately call these wrappers; use them directly for simple local
development or exact control over an existing shell workflow.

```bash
uv run sh scripts/train.sh -c <config-path> [options] [-- --options key=value ...]
```

```{list-table}
:header-rows: 1
:widths: 16 84

* - Flag
  - Meaning
* - `-c CONFIG`
  - config path under `configs/`, with or without `.py`
* - `-n NAME`
  - experiment name (default: `<config-name>-<timestamp>`)
* - `-g GPUS`
  - GPUs per machine; defaults to all visible CUDA devices
* - `-m MACHINES`
  - number of machines; defaults to `1`
* - `-w WEIGHT`
  - checkpoint path passed as `weight=`
* - `-r true`
  - resume from the latest complete checkpoint
* - `-a NAME`
  - W&B run name override
* - `-p PYTHON`
  - Python interpreter
* - `-C`
  - dev mode; skip the code snapshot and run from repository source
```

`train.sh` loads `.env`, snapshots `scripts`, `tools`, and `pimm` into
`exp/<group>/<name>/code/`, and runs from that snapshot (use `-C` for live
edits). Note the direct-script override form is `-- --options key=value …`,
unlike the launchers' bare `-- key=value`.

Testing mirrors it:

```bash
uv run sh scripts/test.sh -c <config-path> -n <experiment-name> [-w model_best]
```

`test.sh` runs `pimm/test.py` with `PYTHONPATH` pointed at the experiment's saved
code snapshot, so a test evaluates the code associated with the experiment, not
your working tree. `-g`/`-m` are parsed for consistency but testing is not
wrapped in `torchrun`.
:::

## Where to go next

- {doc}`concepts` - the ideas the rest of the docs assume (packed tensors, registries, configs).
- {doc}`../configuration/index` - write and inherit Python configs in depth.
- {doc}`../tutorials/byo_dataset_semseg` - bring your own data and train PTv3.
- {doc}`../hpc/index` - move from a node to a managed Slurm cluster.
