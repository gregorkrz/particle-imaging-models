# Core concepts

Four ideas explain almost everything about how pimm is laid out. If you read
only one page before diving in, read this one.

## 1. Packed point clouds with offsets

Detector events are **variable length** — one neutrino interaction might be 100
hits, the next 10,000. Padding to a fixed size would be wasteful and wrong. So
pimm batches point clouds **packed**: every batched quantity is 2D `(N, C)`
instead of 3D `(B, N, C)`, and an `offset` tensor marks where each event ends.

```text
event 0: 100 pts ┐
event 1: 150 pts ├─▶  coord:  (330, 3)   feat: (330, C)
event 2:  80 pts ┘     offset: [100, 250, 330]   # cumulative
```

`offset` is the cumulative sum of per-event lengths — conceptually the same as
PyG's `batch` vector (and {py:class}`~pimm.models.utils.structure.Point` derives one from the other). A model-facing
batch usually looks like:

```python
{
    "coord":  Tensor[total_points, D],   # D = 2 or 3
    "feat":   Tensor[total_points, C],   # features fed to the backbone
    "offset": Tensor[batch_size],        # cumulative event boundaries
    "segment": Tensor[total_points],     # per-point labels (when supervised)
    "name":   list[str],                 # event identifiers
}
```

The collate function **concatenates** tensors (it does not stack), turns
per-sample lengths into cumulative offsets, and skips any key whose name starts
with `_`. This packed contract is what makes the same models work across very
different detectors. See {doc}`../datasets/packed_format`.

## 2. Everything is built from a registry

Models, datasets, transforms, hooks, losses, optimizers, schedulers, trainers,
and testers are all created from **config dictionaries with a `type` key**,
resolved through small registries:

```python
model = build_model(dict(
    type="DefaultSegmentorV2",
    num_classes=5,
    backbone=dict(type="PT-v3m2", in_channels=4),
    criteria=[dict(type="CrossEntropyLoss", ignore_index=-1)],
))
```

`type` is the name a class registered under via a decorator:

```python
@MODELS.register_module("PT-v3m2")
class PointTransformerV3(PointModule):
    ...
```

:::{important}
**Registration is by import side effect.** A class that is defined but never
imported is *not* buildable from config. New models/datasets/hooks must be
imported in the relevant package `__init__.py` — not just via a config
`__import__` — so they survive a resume (which reloads the dumped config). This
is the single most common "why can't it find my class?" gotcha.
:::

The registries:

| Registry | Lives in | Builds |
|----------|----------|--------|
| `MODELS` | `pimm/models/builder.py` | models & backbones |
| `DATASETS` | `pimm/datasets/builder.py` | datasets |
| `TRANSFORMS` | `pimm/datasets/transform/common.py` | transforms |
| `HOOKS` | `pimm/engines/hooks/builder.py` | training hooks |
| `LOSSES` | `pimm/models/losses/builder.py` | loss functions |
| `TRAINERS` | `pimm/engines/train.py` | trainers |

## 3. Configs are Python, execution is YAML

There are two configuration systems and they own different things:

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

Python configs are *real Python* — `_base_` inheritance, list comprehensions
for layer-wise learning rates, values derived from earlier entries. Keep Slurm
accounts and site paths **out** of Python configs; keep model architecture
**out** of launch YAML. See {doc}`../configuration/index`.

## 4. One trainer, one forward contract

The `DefaultTrainer` loop is small and predictable. For each step it moves the
batch to the device, calls the model, and reads one key:

```python
output_dict = model(input_dict)
loss = output_dict["loss"]          # scalar tensor → backward
```

That is the whole contract for training with `DefaultTrainer`: **return a dict
with a scalar `loss`**. Internal modules and backbones are free to return
`Point`, tensors, or tuples. Evaluators and hooks consume a few more
conventional keys when present:

| Key | Consumed by |
|-----|-------------|
| `loss` | trainer (backward + logging) |
| `seg_logits` / `sem_logits` | semantic-seg evaluators & testers |
| `cls_logits` | classification |
| `point` | instance/panoptic evaluators (`return_point=True`) |
| `pred_logits`, `pred_masks`, `pred_momentum` | detector / instance outputs |
| `total_loss` | logging hooks (preferred over raw `loss` when present) |

See {doc}`../models/index` and {doc}`../hooks/index`.

## How the pieces connect

```text
config.py ──build──▶ model ┐
          ──build──▶ dataset ─▶ Compose(transform) ─▶ collate ─▶ packed batch
          ──build──▶ hooks ┐                                        │
          ──build──▶ optimizer / scheduler                          ▼
                                                   DefaultTrainer.run_step()
                                                   ├─ move batch to device
                                                   ├─ output = model(batch)
                                                   ├─ loss = output["loss"]
                                                   ├─ backward / step / sched
                                                   └─ hooks: log, eval, checkpoint
```

Each arrow is a registry build; each `hooks` entry plugs into the lifecycle
(`before_train`, `before/after_step`, `after_epoch`, ...). The trainer itself
stays generic — task specifics live in the model's `forward` and in hooks.

## Glossary

```{glossary}
packed tensor
  A batched quantity stored as 2D `(N, C)` with an `offset` vector instead of a
  padded 3D `(B, N, C)` tensor.

offset
  Cumulative per-event point counts; `offset[i]` is the end index of event `i`
  in the packed tensors. Equivalent to PyG's `batch`.

Point
  An `addict.Dict` (`pimm/models/utils/structure.py`) holding point-cloud state
  — `coord`, `feat`, `offset`/`batch`, serialized orders, sparse-conv state —
  passed between `PointModule`s.

registry
  A name → class table (e.g. `MODELS`) populated by `@register_module`
  decorators at import time and used by `build_*` to construct objects from
  config dicts.

config group
  The directory part of a config path (e.g. `panda/pretrain`), used to place
  the experiment under `exp/<config-group>/<name>/`.

standard checkpoint
  The default split layout: `model/last/weights.pth` (portable model weights) +
  `model/last/trainer.dcp/` (optimizer/RNG/dataloader as a reshardable DCP).
```

## Next

- {doc}`../configuration/index` — Python configs in depth.
- {doc}`../datasets/index` — datasets, transforms, and the packed contract.
- {doc}`../distributed/index` — how this scales to many GPUs and nodes.
