# Your first runs

The fastest way to understand pimm is to run it. This page walks through a
sequence of **real commands** — each one teaches a piece of how pimm works and,
together, they confirm your setup is healthy end to end. Run them in order from
the repository root, in your pimm environment (the conda env or a bound
container; see {doc}`installation`).

Every step shows the exact command and what you should see. The whole sequence
uses one config — the 5-class PILArNet-M semantic segmentation fine-tune — so you
only learn one set of paths:

```text
panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-enc-upcast-fft
```

:::{tip}
Every command here is safe and small. Steps 1, 2, and 6 don't train at all (they
print help or *render* what would run), and step 3 trains for well under a minute
on a few events.
:::

## 1. See what the launcher can do

```bash
pimm launch --help
```

**What you should see.** A Tyro-generated help screen with nested, dotted flags
grouped into sections (`options`, `paths`, `resources`, `slurm`, `run`,
`train`, …):

```text
usage: pimm launch [-h] [OPTIONS]
  --dry-run, --no-dry-run            (default: False)
  --resources.nproc-per-node INT     (default: 1)
  --run.name {None}|STR              (default: None)
  --run.timestamp, --run.no-timestamp
  --train.config STR
  --train.weight {None}|STR
  ...
```

These are the knobs you'll use below. There are no flat `--config` / `--set`
flags — everything is dotted (`--train.config`, `--resources.nproc-per-node`).
{doc}`../reference/cli` documents the full surface.

## 2. Render a command without running it

`--dry-run` resolves the config, run name, and resources and prints the exact
`torchrun`/`scripts/train.sh` invocation it *would* run — without touching a GPU.
Get in the habit of doing this before anything expensive.

```bash
pimm launch \
  --train.config panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-enc-upcast-fft \
  --dry-run
```

**What you should see.** A short shell script ending in the rendered
`scripts/train.sh` call. The config path, machine/GPU counts (`-m 1 -g 1`), and
auto-generated run name are all visible — nothing trains:

```text
sh ./scripts/train.sh -m 1 -g 1 \
  -c panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-enc-upcast-fft \
  -n semseg-pt-v3m2-pilarnet-ft-5cls-enc-upcast-fft-<timestamp> ...
```

## 3. Train a real model (quickly)

Now actually train — but with tiny `max_len` limits, no workers, and no W&B, so
the *entire* pipeline (dataset → transforms → packed batch → model → loss →
backward → checkpoint) runs in seconds. This is the step that proves your
install works.

```bash
pimm launch \
  --train.config panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-enc-upcast-fft \
  --run.name first-run --run.no-timestamp \
  -- epoch=1 data.train.max_len=64 data.val.max_len=32 \
     batch_size=4 num_worker=0 use_wandb=False evaluate=False
```

Everything after the bare `--` is a training override (`KEY=VALUE`). Two of them
keep this first run fast and self-contained:

- `--run.no-timestamp` writes to exactly `exp/panda/semseg/first-run/` instead of
  appending a launch timestamp to the run name (the default).
- `evaluate=False` skips the post-training test pass — the full test split is
  large, and a 16-step run never saves a "best" checkpoint to test. You're
  confirming that training and checkpointing work, not measuring accuracy.

**What you should see.** Per-step loss lines, a final train summary, and a saved
checkpoint:

```text
Train: [1/1][16/16] ... loss: 0.9232 avg_pts: 2270.0000 Lr: 0.00002
Train result: loss: 1.7586 avg_pts: 2930.7812
Saving checkpoint to: ./exp/panda/semseg/first-run/model/last (weights.pth + trainer/ DCP)
Skipping final evaluation (evaluate=False)
```

The exact loss numbers don't matter — what matters is that the steps run and a
checkpoint lands under `exp/panda/semseg/first-run/`. If this completes, your
CUDA / sparse backends / dataset path all work.

:::{note}
No GPU on the machine where you typed this? The launcher will start but
`torchrun` exits with `RuntimeError: no CUDA devices available`. Run it on a GPU
node — locally, inside a Slurm allocation, or via {doc}`../hpc/index`. No
PILArNet-M data yet? See step 7 to fetch some, or point `PILARNET_DATA_ROOT_V2`
at your copy (see {doc}`installation`).
:::

## 4. Look at what the run produced

```bash
ls exp/panda/semseg/first-run/
tail exp/panda/semseg/first-run/train.log
```

**What you should see.** The launcher snapshots the code and writes the resolved
config and metadata next to the log — the "reproducible by construction" idea in
practice:

```text
code/                   # snapshot pimm/scripts/tools the run trained from
config.py               # fully resolved config (inheritance + overrides + seed)
resolved_config.json    # full resolved config as JSON
model_config.json       # cfg.model as JSON
run_metadata.json       # command, host, git status, original config path
train.log               # the training log you just tailed
model/last/             # the saved checkpoint (weights.pth + trainer/ DCP)
```

Treat `config.py` and `run_metadata.json` as the authoritative record of what the
run started with. With `use_wandb=False` you'll also see a TensorBoard
`events.out.tfevents.*` file here.

## 5. Export the model and load it back

Turn the run's checkpoint into a portable, config-free artifact, then load it in
Python — the path you'd use for inference or sharing.

```bash
pimm export --run-dir exp/panda/semseg/first-run last ./artifacts/first-run
```

**What you should see.** `Exported pretrained model to: artifacts/first-run`, and
the directory containing `model.safetensors` plus `training_config.json`.

Now load it back:

```python
import pimm

model = pimm.from_pretrained("artifacts/first-run")   # ready for inference
print(type(model).__name__)   # -> DefaultSegmentorV2
```

If it prints the model class without error, your export → load round-trip works.
See {doc}`../checkpoints/export` and {doc}`../models/index`.

## 6. Preview a Slurm submission

You don't need a cluster to *see* how a batch job would be composed.
`pimm submit --dry-run` prints the authoritative submitit manifest — check the
resources, account, and partition before anything hits the queue.

```bash
pimm submit --dry-run --site s3df \
  --resources.nnodes 1 --resources.nproc-per-node 4 --resources.time 00:30:00 \
  --train.config panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-enc-upcast-fft
```

**What you should see.** A single YAML manifest describing the job — resolved
account / partition / GRES from the `s3df` site profile and a pre-rendered
`attempts` list with the container `srun ... scripts/train.sh` command:

```text
  slurm_partition: ampere
  slurm_gres: gpu:4
attempts:
- job_index: 1
  run_name: semseg-pt-v3m2-pilarnet-ft-5cls-enc-upcast-fft-<timestamp>
  script: |
    ... singularity run ... sh /opt/pimm/src/scripts/train.sh -m 1 -g 4 -c ...
```

`pimm launch` errors if the resolved site is non-local — use `pimm submit` for
Slurm. Swap `--site s3df` for `--site nersc` to compare. Full details:
{doc}`../hpc/index`.

## 7. Optional extras

::::{tab-set}

:::{tab-item} Get real data
The quick run above uses tiny limits. To train on real PILArNet-M data, download
a revision and point the matching env var at it:

```bash
python scripts/download_pilarnet.py --version v2 --output-dir /path/to/pilarnet
export PILARNET_DATA_ROOT_V2=/path/to/pilarnet/v2
```

`PILArNetH5Dataset` reads `PILARNET_DATA_ROOT_V1` / `_V2` / `_V3` (and falls back
to `~/.cache/pimm/pilarnet/<revision>`). See {doc}`../datasets/pilarnet`.
:::

:::{tab-item} Log to Weights & Biases
Rerun step 3 with W&B logging once you've authenticated:

```bash
wandb login                      # or: export WANDB_MODE=offline
pimm launch \
  --train.config panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-enc-upcast-fft \
  --run.name first-run-wandb \
  -- epoch=1 data.train.max_len=64 data.val.max_len=32 \
     batch_size=4 num_worker=0 use_wandb=True evaluate=False
```

Rank 0 streams metrics to W&B; with `use_wandb=False` (step 3) it writes
TensorBoard events instead. See {doc}`../hooks/logging`.
:::

:::{tab-item} Preview a warm-start
Fine-tuning configs can warm-start from a published checkpoint. Preview the
rendered command — `--train.weight` accepts an `hf://` URI:

```bash
pimm launch --dry-run \
  --train.config panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-enc-upcast-fft \
  --train.weight hf://youngsm/sonata-pilarnet-L/model_best.pth
```

The rendered `scripts/train.sh` call gains a `-w hf://...` argument. See
{doc}`../checkpoints/huggingface`.
:::

::::

## Where to go next

- {doc}`quickstart` — the launcher flags and override mechanics in depth.
- {doc}`../tutorials/byo_dataset_semseg` — wrap your own detector data and train a
  PTv3 semantic-segmentation model from a registered dataset.
- {doc}`../hpc/index` — take it to a real cluster.
</content>
