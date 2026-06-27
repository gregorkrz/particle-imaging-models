# Core concepts

This page includes a few of the more important concepts in pimm that may differ from other ML libraries or your past conventions.

## 1. Packed point clouds with offsets

This repository is set up to work with sparse data (e.g., point clouds) dictionaries.

In high energy physics, events are **variable length** — one might be 100 hits or particles, the next 10,000. In machine learning, we often want to train on many events at a single iteration via stochastic gradient descent. Oftentimes this is done by **padding** individual events into a 3D tensor, so that a model can ingest and work on events in parallel. Having this 3D tensor of shape `(B, N, C)` where `B` is the batch size, `N` is your expected maximum number of data points per event (i.e., what each event is padded to), and `C` are data-level features (e.g., energy deposited), wastes quite a lot of memory, especially if the amount of data in each event varies widely.  So pimm batches data in a **packed** format: every batched quantity is 2D `(N, C)` instead of 3D `(B, N, C)`, and an `offset` tensor marks where each event ends. This essentially is the concatenation of all data into a single flat 2D tensor.  This data format is called Compressed Sparse Row (CSR). 

Graph neural networks, for example, make heavy use of this data format, as they naturally work with variable length objects. One downside is that your memory usage is no longer predictable, and is a bit more spiky as it is [fragmented](https://en.wikipedia.org/wiki/Fragmentation_(computing)). This means that you can hit Out of Memory (OOM) errors halfway through a training run if your batch contains more data than normal -- so be careful! I would advise logging GPU VRAM usage over the course of training with the {py:class}`~pimm.engines.hooks.resources.ResourceUtilizationLogger` hook.

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

Note that some methods have fancier data formats, like the {py:class}`~pimm.models.utils.structure.Point` class, which is used in Point Transformer V3 to automatically deal with setting the correct backend formats for sparse CNN libraries.

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

This makes these objects very customizable and modular, and makes things quite easy to extend, i.e. adding a new model(link), adding a new dataset(link), adding a new hook(link), adding a new trainer(link). An example where this is extremely helpful is in foundation model research, where you are researching both what model encoder to use (Sparse UResNet? GNN? Point Transformer V3?) and what training paradigm to try (MAE? Sonata? JEPA?). If you set up your models in a smart way, registries allow these two R&D axes to be mostly independent from one another. For example, Sonata just needs a feature extractor to give it per-point features; it doesn't care about how you give those features. E.g., the Sonata config looks something like:

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

| Registry | Lives in | Builds (all registered types) |
|----------|----------|--------|
| `MODELS` | `pimm/models/builder.py` | {doc}`models & backbones <../api/registry/models>` |
| `DATASETS` | `pimm/datasets/builder.py` | {doc}`datasets <../api/registry/datasets>` |
| `TRANSFORMS` | `pimm/datasets/transform/common.py` | {doc}`transforms <../api/registry/transforms>` |
| `HOOKS` | `pimm/engines/hooks/builder.py` | {doc}`training hooks <../api/registry/hooks>` |
| `LOSSES` | `pimm/models/losses/builder.py` | {doc}`loss functions <../api/registry/losses>` |
| `TRAINERS` | `pimm/engines/train.py` | {doc}`trainers <../api/registry/trainers>` |

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
# launch/sites/nersc.yaml
_base_: slurm.yaml

site: nersc

paths:
  repo_root: /global/homes/y/youngsam/sw/pimm-private
  exp_root: /global/homes/y/youngsam/data

resources:
  cpus_per_proc: 6
  mem: 192G

slurm:
  account: m5238_g
  qos: regular
  constraint: gpu
  gpu_directive: gpus-per-node
  image: youngsm/pimm-nersc:main
  module: gpu,nccl-plugin

container:
  runtime: shifter
  image: youngsm/pimm-nersc:main
  module: gpu,nccl-plugin
  unset_env:
    - LD_LIBRARY_PATH
    - LD_PRELOAD
  setup:
    - export LD_LIBRARY_PATH=/opt/hdf5/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

env:
  # Perlmutter compute nodes reach huggingface.co (HEAD/redirect OK) but the
  # Xet CAS transfer endpoint (cas-bridge.xethub.hf.co) stalls -> downloads sit
  # at 0 MB. Force the classic LFS/HTTPS path via cdn-lfs, which works in-job.
  HF_HUB_DISABLE_XET: "1"
  NCCL_SOCKET_IFNAME: "^docker0,lo"
  NCCL_NET_GDR_LEVEL: PHB
  HDF5_USE_FILE_LOCKING: "FALSE"
```
:::

::::

A run command using this config and site configuration would be:
```sh
pimm submit --site nersc --train.config path/to/some_config
```

There's also some more features, like run YAMLs, but read more in 

## 4. Models being trained must output a loss

Like most ML training libraries, training involves around a Trainer (link to DefaultTrainer in API), which sets up an entire run, runs the training loop (link to https://www.deeplearnphysics.org/particle-imaging-models/stable/_modules/pimm/engines/train.html#Trainer.train), including all hooks,
and cleans up once everything is finished. For each step, our it moves the batch to the device, calls the model, and reads one key:

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
Per-step metrics in the model outputs here are not automatically synchronized across GPUs. [Distributed Data Parallel (DDP)](https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html) training (what we use to do multi-GPU, multi-machine training) all-reduces gradients from `loss` during the backward(), so training is correct, but the scalar values in the model's output dict (including loss) are each rank's local values, and the logging hooks will usually record them on the main process only. If you compute a metric in a custom hook and want the true global value, you will need to all-reduce it yourself. Some models, e.g. Sonata, already all-reduce their component/total_loss values for logging, but intentionally leave the autograd loss key local:

```python
import torch.distributed as dist

def forward(self, data_dict, return_point=False):
  ...
  # sync component losses for logging
  if (ws:=get_world_size()) > 1:
      for key in list(output_dict.keys()):
          if key == 'loss':
              continue
          synced_loss = output_dict[key].detach()
          dist.all_reduce(synced_loss, op=dist.ReduceOp.SUM)
          synced_loss.div_(ws)
          output_dict[key] = synced_loss
  return output_dict
```
:::




## Next

- {doc}`../configuration/index` — Python configs in depth.
- {doc}`../datasets/index` — datasets, transforms, and the packed contract.
- {doc}`../distributed/index` — how this scales to many GPUs and nodes.
