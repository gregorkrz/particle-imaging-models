# How pimm works

{doc}`overview` introduces pimm; this material builds on it.

This page includes a few of the more important concepts in pimm that may differ from other ML libraries or your past conventions.

## 1. Packed point clouds with offsets

This repository is set up to work with sparse data (e.g., point clouds) dictionaries.

In high energy physics, events are **variable length** - one might be 100 hits or particles, the next 10,000.
Training on many events per iteration is often done by **padding** individual events into a 3D tensor of shape `(B, N, C)`, where `B` is the batch size, `N` is the padded maximum number of points per event, and `C` are per-point features (e.g., energy deposited).
This wastes memory when event sizes vary widely, so pimm instead batches data in a **packed** format: every batched quantity is 2D `(N, C)`, the concatenation of all events into a single flat tensor, with an `offset` tensor marking where each event ends.
This layout is Compressed Sparse Row (CSR), and graph neural networks make heavy use of it since they work naturally with variable length objects.
Packed memory usage is not predictable in advance, so a batch with more data than normal can trigger Out of Memory (OOM) errors partway through a run; log GPU VRAM usage across training with the {py:class}`~pimm.engines.hooks.resources.ResourceUtilizationLogger` hook.

<p align="center">
    <img alt="pointcept" class="only-light" src="https://raw.githubusercontent.com/pointcept/assets/main/pointcept/offset.png" width="480">
    <img alt="pointcept" class="only-dark" src="https://raw.githubusercontent.com/pointcept/assets/main/pointcept/offset_dark.png" width="480">
    <br>
</p>


A batch in pimm usually looks something like:

```python
{
    "coord":  Tensor[total_points, D],   # D = 2 or 3, and correspond to 3D positions
    "feat":   Tensor[total_points, C],   # features fed as inputs to the model
    "offset": Tensor[batch_size],        # cumulative event boundaries
    "segment": Tensor[total_points, 1],  # per-point labels (when supervised)
    "name":   list[str],                 # event identifiers
}
```

A collate function automatically **concatenates** tensors into this flat format, turning
per-sample lengths into cumulative offsets. This packed format is quite general, and what makes the same models work across very
different detectors. See {doc}`../datasets/data_format`.

## 2. Everything is a registry

Models, datasets, transforms, hooks, losses, optimizers, schedulers, trainers,
and testers are all created from **dictionaries with a `type` key**,
resolved through small registries, e.g.:

```python
from pimm.models import build_model
model = build_model(dict(type="PT-v3m2", in_channels=3, ...))
```

`type` is the name a class registered under via a decorator placed in front of all datasets, transforms, hooks, models, and trainers:

```python
from pimm.models import MODELS

@MODELS.register_module("PT-v3m2")      # <-- type would be "PT-v3m2"
class PointTransformerV3(PointModule):
    ...
```

This makes these objects modular and easy to extend: see {doc}`../research_ecosystem/contributing_a_model`, {doc}`../research_ecosystem/contributing_a_dataset`, {doc}`../research_ecosystem/contributing_a_hook`, and {doc}`../research_ecosystem/contributing_a_transform`.
Registries are helpful in foundation model research, where the model encoder (Sparse UResNet, GNN, Point Transformer V3) and the training paradigm (MAE, Sonata, JEPA) are separate axes of study.
Registries keep these two axes mostly independent: Sonata requires only a feature extractor that produces per-point features, and how those features are produced does not matter.
The Sonata config looks something like:

```python
model = dict(
  type="Sonata-v1m1",
  backbone=dict(
    type="PT-v3m2",
    in_channels=4,
    ...
  ),
  head_in_channels=512, # number of features output by the backbone
  ...
)
```
Look around some of the configuration files in pimm to get an idea of this.

:::{important}
A class that pimm never imports doesn't get registered, so it isn't buildable from a config. Import new models/datasets/transforms/hooks from the relevant package `__init__.py`.
:::

The registries below can be found in the {doc}`API reference <../api/index>`.

| Registry | Builds (all registered types) |
|----------|--------|
| `MODELS` | {doc}`models & backbones <../api/registry/models>` |
| `DATASETS` | {doc}`datasets <../api/registry/datasets>` |
| `TRANSFORMS` | {doc}`transforms <../api/registry/transforms>` |
| `HOOKS` | {doc}`training hooks <../api/registry/hooks>` |
| `LOSSES` | {doc}`loss functions <../api/registry/losses>` |
| `TRAINERS` | {doc}`trainers <../api/registry/trainers>` |

## 3. Training configs are Python, command executions are in YAML

There are two configuration systems and they do different things:

```{list-table}
:header-rows: 1
:widths: 22 78

* - Layer
  - Owns
* - **Python configs** (`configs/*.py`)
  - *What* to train: model, dataset, transforms, optimizer, scheduler, hooks,
    epochs, batch size. The source of truth for training behavior.
* - **Launch YAML** (`launch/`)
  - *How / where* to run: Slurm resources, account, partition, container,
    site paths, env vars, run naming, resume, chaining.
```

Example configs:

::::{tab-set}

:::{tab-item} Python config
```python
# configs/path/to/some_config.py
_base_ = ["../../_base_/default_runtime.py"]

# Runtime and logging
batch_size = 48       # batch size / GPU
num_worker = 24       # num worker / GPU
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

Hooks and tester overrides
hooks = [...]
test = dict(type="...", ...)
```
:::

:::{tab-item} YAML
```yaml
# launch/sites/mycluster.yaml
_base_: slurm.yaml                 # generic Slurm defaults

site: mycluster

paths:
  repo_root: /path/to/pimm         # shared checkout jobs run from
  exp_root: "{repo_root}/exp"      # where runs are written

resources:
  nnodes: 1
  nproc_per_node: 4                # GPUs per node
  cpus_per_proc: 12                # CPUs per GPU
  time: "12:00:00"

slurm:
  account: <account>
  partition: <partition>
  gpu_directive: gres              # `--gres=gpu:N`; some clusters need `gpus-per-node`

container:
  runtime: none                    # or `singularity` with an `image:`

env:
  NCCL_SOCKET_IFNAME: "^docker0,lo"
  HDF5_USE_FILE_LOCKING: "FALSE"
```
:::

::::

A run command using this config and site configuration would be:
```sh
pimm submit --site mycluster --train.config path/to/some_config
```

There are more features, like run YAMLs; read more in {doc}`../hpc/index`.

## 4. Models being trained must output a loss

Like most ML training libraries, training centers on a {doc}`Trainer <../api/registry/trainers>`, which sets up a run, runs the training loop including all hooks, and cleans up once everything is finished.
For each step, it moves the batch to the device, calls the model, and reads one key:

```python
for input_dict in data_loader:
    ...
    output_dict = model(input_dict)
    loss = output_dict["loss"]      # <-- all models being trained need this!
    ...
    loss.backward()
```

Internal modules and backbones are free to return `Point`, tensors, etc. Different evaluators and hooks, when used, assume a few more
keys are in `output_dict`:

| Key | Consumed by |
|-----|-------------|
| `loss` | trainer (backward + logging) |
| `seg_logits` / `sem_logits` | semantic-seg evaluators & testers |
| `cls_logits` | classification |
| `point` | instance/panoptic evaluators (using `return_point=True`) |
| `pred_logits`, `pred_masks`, `pred_momentum` | detector / instance outputs |
| `total_loss` | logging hooks (preferred over raw `loss` when present) |

See {doc}`../research_ecosystem/using_trained_models` and {doc}`../hooks/index`.

:::{note}
In multi-GPU training, [Distributed Data Parallel (DDP)](https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html) synchronizes gradients during `backward()`, so training is correct.
The logged per-step scalars in the output dict (including `loss`) are each rank's local values unless a model all-reduces them itself.
:::




## Next

- {doc}`../configuration/index` - Python configs in depth.
- {doc}`../datasets/index` - datasets, transforms, and the packed format.
- {doc}`../distributed/index` - how this scales to many GPUs and nodes.
