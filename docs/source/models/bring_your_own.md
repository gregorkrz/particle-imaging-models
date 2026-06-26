# Bring your own model

This page is a practical walkthrough for adding a **new model** to pimm so the
trainer, evaluators, checkpointer, and launcher all drive it for free. The
contract is small: a model is a registered `nn.Module` whose
`forward(input_dict)` consumes a {doc}`packed batch <../datasets/packed_format>`
and returns a **dict** — with a scalar `loss` while training.

:::{seealso}
The mirror image of this page is {doc}`../datasets/bring_your_own` (custom
**data**). For the data → model → train story end-to-end, see the tutorial
{doc}`../tutorials/byo_dataset_semseg`. For the mental model (registries, packed
tensors, the trainer/model contract), read {doc}`../getting_started/concepts`.
:::

## 0. Decide how much you need to write

```text
new task head or loss on an existing backbone?  ──▶  reuse DefaultSegmentorV2, write a config
new backbone architecture?                      ──▶  register a backbone, drop it in model.backbone
genuinely new forward / output structure?       ──▶  write a model class (this page)
```

Most additions are **not** a new top-level model. A new backbone plugs straight
into {py:class}`~pimm.models.default.DefaultSegmentorV2` via `model.backbone.type`; a different head or
loss mix is often just a config change. Write a new model class only when the
**forward pass or output structure** is genuinely different (a new task, a
multi-branch loss, a custom inference output). If that's you, read on.

## 1. The contract

A pimm model is an `nn.Module` registered in the `MODELS` registry. The trainer
moves the collated batch to the device and calls your model with it; for
evaluation, evaluators call it with `return_point=True`. So the signature is:

```python
def forward(self, input_dict, return_point=False): ...
```

**What you receive.** `input_dict` is the packed batch (see
{doc}`../datasets/packed_format`) — concatenated across the mini-batch, no batch
dimension. The keys your model can rely on:

| Key | Shape | Notes |
|-----|-------|-------|
| `coord` | `(N, 3)` | point coordinates (post-transform) |
| `grid_coord` | `(N, 3)` | integer voxel coords (sparse backbones need these) |
| `feat` | `(N, C)` | input features; `C` **must** equal `backbone.in_channels` |
| `offset` | `(B,)` | cumulative per-event boundaries; `offset[-1] == N` |
| `segment` / `instance` | `(N,)` | supervision targets (present in train/eval, absent at test) |

Wrap it in a {py:class}`~pimm.models.utils.structure.Point` (`point = Point(input_dict)`) and the
backbone/serialization machinery takes over; `Point` derives `batch` from
`offset` for you.

**What you must return.** A `dict`. The single hard requirement is that during
training it contains a scalar **`loss`** — the trainer backpropagates
`output_dict["loss"]` and nothing else. The idiomatic three-branch return
(train / eval / test) also surfaces the logits evaluators consume:

```python
# train  → {"loss": scalar}
# eval   → {"loss": scalar, "seg_logits": (N, num_classes)}   # target present
# test   → {"seg_logits": (N, num_classes)}                   # no target
```

:::{tip}
**`loss` vs `total_loss`.** The trainer only ever reads `loss`. `total_loss`, if
you return it, is a *detached* logging alias — the logging hook displays it under
the name "loss" instead of the graph-attached tensor. Return it only if you want
the logged number to differ from the backpropped one; otherwise just return
`loss`.
:::

## 2. Write the model

This skeleton is a faithful, minimal copy of
{py:class}`~pimm.models.default.DefaultSegmentorV2` — the canonical pattern of *backbone → head →
criteria*. Start from it and change the head, the targets, or the output keys.

```python
# pimm/models/my_model/my_segmentor.py
import torch
import torch.nn as nn

from pimm.models.builder import MODELS, build_model
from pimm.models.losses import build_criteria
from pimm.models.utils.structure import Point


@MODELS.register_module("MySegmentor")
class MySegmentor(nn.Module):
    def __init__(self, num_classes, backbone_out_channels,
                 backbone=None, criteria=None, freeze_backbone=False):
        super().__init__()
        self.backbone = build_model(backbone)          # a registered backbone, built from its dict
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0 else nn.Identity()
        )
        self.criteria = build_criteria(criteria)        # list[dict] -> callable(pred, target)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def forward(self, input_dict, return_point=False):
        point = self.backbone(Point(input_dict))

        # Backbones return a (possibly pooled) Point; unwrap to per-point feats.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point

        seg_logits = self.seg_head(feat)               # (N, num_classes)

        out = {}
        if return_point:                               # instance/PCA evaluators read point
            out["point"] = point
        if self.training:                              # TRAIN: must return a scalar loss
            out["loss"] = self.criteria(seg_logits, input_dict["segment"])
        elif "segment" in input_dict:                  # EVAL: loss + logits
            out["loss"] = self.criteria(seg_logits, input_dict["segment"])
            out["seg_logits"] = seg_logits
        else:                                          # TEST: logits only
            out["seg_logits"] = seg_logits
        return out
```

:::{important}
`backbone_out_channels` must equal the backbone's final feature width — e.g. a
`PT-v3m2` with a decoder ends at `dec_channels[0]` (`64` in the reference
config), while the same backbone in `enc_mode=True` (encoder-only) ends wider
(`1232`). Mismatch is the most common "shapes don't line up" error. Likewise
`feat`'s channel count (set by `Collect(feat_keys=...)`) must equal
`backbone.in_channels`.
:::

## 3. Wire in losses

You don't compute the loss by hand — declare it in the config as a list of loss
dicts and let `build_criteria` assemble them. The resulting callable is
`criteria(pred, target)` and it **sums** every loss in the list:

```python
criteria=[
    dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
    dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0 / 20.0, ignore_index=-1),
]
```

Each entry is built from the `LOSSES` registry (see the {doc}`API reference <../api/index>`). A
loss's `forward(pred, target)` takes `pred` of shape `(N, C)` and `target` of
shape `(N,)`. Need a loss that doesn't exist yet? Register one the same way you
register a model — `@LOSSES.register_module()` — and reference it by `type`.

## 4. Register it (import side effect)

The `@MODELS.register_module()` decorator only runs when the module is imported.
Add it to `pimm/models/__init__.py`:

```python
# pimm/models/__init__.py
from .my_model.my_segmentor import MySegmentor   # noqa: F401
```

:::{warning}
Importing your model only from a config's `__import__` is **not enough**: resume
replays the *dumped* config, which has no `__import__`, so `MODELS.build(...)`
won't find your class and the run dies on restart. Register it in the package
`__init__.py`. This is the single most common "why can't it find my model?"
gotcha — see {doc}`../getting_started/concepts`.
:::

## 5. Write a config

Select your model by `type` and pass its constructor kwargs, including the
backbone (itself a registered `type`) and the criteria list:

```python
model = dict(
    type="MySegmentor",
    num_classes=5,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m2",
        in_channels=4,            # xyz + energy → must match Collect(feat_keys=...)
        # ... backbone dims; crib from configs/panda/semseg/...-fft.py
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0 / 20.0, ignore_index=-1),
    ],
)
```

See {doc}`../configuration/index` for `_base_` inheritance and overrides, and
{doc}`../reference/model_zoo` for the registered backbone/head `type` names.

## 6. Verify forward + loss

Before launching, confirm your model builds and produces a backprop-able loss on
a synthetic packed batch — no dataset required (this needs the GPU stack, since
`PT-v3*` backbones use spconv/CUDA):

```python
import torch
from pimm.models.builder import build_model

model = build_model(dict(
    type="MySegmentor", num_classes=5, backbone_out_channels=64,
    backbone=dict(type="PT-v3m2", in_channels=4),       # + the dims from your config
    criteria=[dict(type="CrossEntropyLoss", ignore_index=-1)],
)).cuda().train()

N = 4000
batch = dict(                                            # two events: offset = [N//2, N]
    coord=torch.rand(N, 3).cuda(),
    grid_coord=torch.randint(0, 256, (N, 3)).cuda(),
    feat=torch.rand(N, 4).cuda(),                        # C == backbone.in_channels
    offset=torch.tensor([N // 2, N]).cuda(),
    segment=torch.randint(0, 5, (N,)).cuda(),
)

out = model(batch)
assert "loss" in out and out["loss"].requires_grad      # the trainer backprops this
out["loss"].backward()                                  # gradients flow end-to-end
print(float(out["loss"]))
```

If `loss` is a scalar with `requires_grad=True` and `backward()` runs, the
trainer can drive your model.

## 7. What you get for free

Once your model is a registered `nn.Module` with the contract above, you inherit
the whole stack — no extra wiring:

- **Distributed** — DDP and FSDP2 wrap any `nn.Module` ({doc}`../distributed/index`).
- **Checkpoint + exact resume** — model/optimizer/RNG/dataloader state
  ({doc}`../checkpoints/index`, {doc}`../hpc/resuming`).
- **Export + Hub** — `pimm export` / {py:func}`~pimm.from_pretrained` round-trips it
  ({doc}`../checkpoints/export`, {doc}`../checkpoints/huggingface`).
- **Hooks + evaluators + HPC launch** — the full lifecycle ({doc}`../hooks/index`,
  {doc}`../hpc/index`).

A few caveats worth knowing:

- **Accept `return_point`** (even if unused) so instance/PCA evaluators, which
  call `model(input_dict, return_point=True)`, don't error.
- **DDP uses `static_graph=True`.** If your model uses a different set of
  parameters step-to-step (data-dependent branches, conditionally-used heads),
  set `find_unused_parameters=True` in the config.
- **The training return must contain `loss`** (a `KeyError` otherwise), and the
  eval/test branches must emit whatever logits your evaluator reads (next table).

## Output keys per task

The keys evaluators and testers look for (return the ones your task needs):

```{list-table}
:header-rows: 1
:widths: 30 70

* - Key
  - Consumed by
* - `loss`
  - the trainer (backprop), training mode — **required**
* - `seg_logits` (or `sem_logits`)
  - {py:class}`~pimm.engines.hooks.eval.semantic_segmentation.SemSegEvaluator` / `SemSegTester` — per-point class logits
* - `point`, `pred_masks`, `pred_logits` (+ `pred_momentum`)
  - {py:class}`~pimm.engines.hooks.eval.instance_segmentation.InstanceSegmentationEvaluator` (call with `return_point=True`, val batch size 1)
* - `cls_logits`
  - classification convention (see {py:class}`~pimm.models.default.DefaultClassifier`)
* - `total_loss`
  - logging only (detached); optional
```

## Checklist

1. Subclass `nn.Module`; register with `@MODELS.register_module("MyModel")`.
2. `forward(self, input_dict, return_point=False)` → `Point(input_dict)` → backbone → head.
3. Return `{"loss": ...}` in training; add task logits for eval/test.
4. Apply losses via `self.criteria = build_criteria(criteria)`.
5. Import the model module in `pimm/models/__init__.py` (survives resume).
6. Write a config selecting it by `type`, with `backbone` + `criteria`.
7. Verify forward + `loss.backward()` on a synthetic batch.

## See also

- {doc}`../datasets/bring_your_own` — the data-side mirror of this page.
- {doc}`../tutorials/byo_dataset_semseg` — the full end-to-end tutorial.
- {doc}`../getting_started/concepts` — registries, packed tensors, the trainer contract.
- {doc}`../datasets/packed_format` — the exact batch your `forward` receives.
- {doc}`../reference/model_zoo` — registered backbones and heads to build on.
