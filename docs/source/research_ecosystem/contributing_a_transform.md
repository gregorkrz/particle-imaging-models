# Contributing a transform

A transform is one step in the {py:class}`~pimm.datasets.transform.base.Compose` pipeline that turns the **flat numpy
dict** a dataset emits into the **packed tensor batch** a model consumes. You
write a new one whenever the built-in catalog ({doc}`../datasets/transforms`)
can't express the step you need.

There are three broad reasons to reach for a custom transform - they look
similar (a registered callable on a `data_dict`) but sit at very different points
in the pipeline:

```text
preprocessing  ──▶  get raw data into the format: relabel, remap, derive a key
augmentation   ──▶  perturb a sample for regularization: jitter, drop, rotate
method machinery ──▶  build the inputs a training method needs: multi-view, masks
```

:::{seealso}
Read {doc}`../datasets/transforms` first for the pipeline, `index_valid_keys`,
the final `Collect`, and the full catalog of built-ins.
:::

## Requirements

A transform is a class registered in the `TRANSFORMS` registry whose `__call__`
takes the sample `data_dict` and returns it (usually the **same** dict, mutated
in place):

```python
@TRANSFORMS.register_module()
class MyTransform(object):
    def __init__(self, **cfg):     # config kwargs from the transform dict
        ...

    def __call__(self, data_dict):
        # read/modify keys, then return the dict
        return data_dict
```

`Compose` builds each `dict(type="MyTransform", ...)` entry through the registry
and threads one `data_dict` through every step in order. A few rules the runner
relies on:

- **Operate on numpy, not tensors.** Transforms run *before*
  {py:class}`~pimm.datasets.transform.base.ToTensor`, so `coord`, `energy`,
  `segment`, … are numpy arrays. Returning tensors early breaks the numpy-based
  geometry steps downstream.
- **Name the spatial array `coord`.** Geometric transforms find points by the key
  `coord`; labels conventionally end up as `segment` / `instance`.
- **Return the dict.** Most transforms mutate and `return data_dict`. A few return
  a *different* object (e.g. `GridSample(mode="test")` returns a list of fragment
  dicts) - the runner just forwards whatever you return to the next step.
- **Keep it JSON-like in config.** Pass `[dict(type=...), ...]`, never a
  pre-built `Compose`.

## 1. Preprocessing - get raw data into the required format

The first job is often converting raw arrays into the keys/format the rest of the pipeline requires: remapping a label scheme, deriving a key, fixing dtypes. These
sit **early**, before augmentation and subsampling. Example - collapse a raw PDG
code array into contiguous class ids:

```python
import numpy as np
from pimm.datasets.transform.common import TRANSFORMS


@TRANSFORMS.register_module()
class MyPDGToClass(object):
    def __init__(self, mapping, default=-1, src_key="pdg", dst_key="segment"):
        self.mapping = {int(k): int(v) for k, v in mapping.items()}
        self.default = int(default)
        self.src_key, self.dst_key = src_key, dst_key

    def __call__(self, data_dict):
        pdg = data_dict[self.src_key]
        seg = np.full(pdg.shape, self.default, dtype=np.int64)
        for code, cls in self.mapping.items():
            seg[pdg == code] = cls
        data_dict[self.dst_key] = seg
        return data_dict
```

```python
dict(type="MyPDGToClass", mapping={22: 0, 11: 1, 13: 2, 211: 3, 2212: 4}),
```

:::{important}
If your transform **adds a new point-aligned array** (length `N`, must follow
the same points as `coord` through subsampling), register it in
`index_valid_keys` *before* the first subsampling transform - otherwise it keeps
its original length and silently desyncs from `coord`. Add it with the `Update`
transform, or have your transform append to `data_dict["index_valid_keys"]`. See
{doc}`../datasets/transforms`.
:::

## 2. Augmentation - perturb a sample

Augmentations act on `coord` (and other point arrays) for regularization. Keep
randomness inside `__call__` so every sample is drawn independently, and gate it
with a probability `p`. The convention mirrors the built-in geometric transforms:

```python
import numpy as np
from pimm.datasets.transform.common import TRANSFORMS


@TRANSFORMS.register_module()
class RandomEnergyDropout(object):
    """Zero a random fraction of points' energy (a feature-space augmentation)."""

    def __init__(self, ratio=0.1, p=0.5):
        self.ratio, self.p = float(ratio), float(p)

    def __call__(self, data_dict):
        if np.random.rand() > self.p or "energy" not in data_dict:
            return data_dict
        n = data_dict["energy"].shape[0]
        mask = np.random.rand(n) < self.ratio
        data_dict["energy"][mask] = 0.0
        return data_dict
```

Augmentations belong in the **train** pipeline only - the val/test pipeline must
stay deterministic so evaluation is reproducible (this is exactly why inference
reuses the *val* transform; see
[reproducing the pipeline at inference](../datasets/transforms.md#reproducing-the-pipeline-at-inference)).

## 3. Method machinery - build what a training method needs

Some training methods need more than per-point edits: they restructure the sample
itself. Multi-view self-distillation (Sonata/DINO-style) needs **several
independently-augmented views** of the same event; masked autoencoding needs a
mask; contrastive learning needs a positive pair. These transforms typically run
a **sub-pipeline** and write their outputs under prefixed keys the model and
collate function then consume.

{py:class}`~pimm.datasets.transform.multiview.ContrastiveViewsGenerator` is the
minimal pattern - copy the source keys into two sub-dicts, run the same
`Compose` independently on each, and write the results back under `view1_*` /
`view2_*`:

```python
from pimm.datasets.transform.common import TRANSFORMS
from pimm.datasets.transform.base import Compose


@TRANSFORMS.register_module()
class MyTwoViewGenerator(object):
    def __init__(self, view_keys=("coord", "energy"), view_trans_cfg=None):
        self.view_keys = view_keys
        self.view_trans = Compose(view_trans_cfg)   # a nested pipeline of dicts

    def __call__(self, data_dict):
        for i in (1, 2):
            view = {k: data_dict[k].copy() for k in self.view_keys}
            view = self.view_trans(view)            # independent augmentation per view
            for k, v in view.items():
                data_dict[f"view{i}_{k}"] = v
        return data_dict
```

```python
dict(type="MyTwoViewGenerator",
     view_keys=("coord", "energy"),
     view_trans_cfg=[
         dict(type="RandomRotate", angle=[-1, 1], axis="z", p=0.8),
         dict(type="RandomFlip", p=0.5),
         dict(type="GridSample", grid_size=0.001, mode="train", return_grid_coord=True),
     ]),
```

The downstream model reads `view1_coord`, `view2_coord`, … and computes the
self-distillation loss across views. For real, larger examples see
{py:class}`~pimm.datasets.transform.multiview.MultiViewGenerator` and
{py:class}`~pimm.datasets.transform.multiview.MixedScaleGeometryMultiViewGenerator`,
and the SSL mask/collate transforms in `pimm/datasets/transform/hmae.py`.

## Register it

Import your transform module from `pimm/datasets/transform/__init__.py` so the
`@TRANSFORMS.register_module()` decorator runs:

```python
# pimm/datasets/transform/__init__.py
from .my_transforms import MyPDGToClass, RandomEnergyDropout, MyTwoViewGenerator  # noqa: F401
```

## Verify it in isolation

A transform is a plain callable on a numpy dict, so you can exercise it without a
dataset or GPU:

```python
import numpy as np
from pimm.datasets.transform import Compose

pipeline = Compose([
    dict(type="MyPDGToClass", mapping={22: 0, 11: 1}),
    dict(type="RandomEnergyDropout", ratio=0.2, p=1.0),
])

event = {"coord": np.random.rand(100, 3).astype("f4"),
         "energy": np.random.rand(100, 1).astype("f4"),
         "pdg": np.random.choice([22, 11, 13], size=100)}

out = pipeline(event)
assert out["segment"].shape == (100,)
assert (out["energy"] == 0.0).any()
```

## Checklist

1. Register the class with `@TRANSFORMS.register_module()`.
2. `__call__(self, data_dict)` reads numpy keys and returns the dict.
3. New point-aligned keys → add to `index_valid_keys` before subsampling.
4. Augmentations go in the **train** pipeline only; val/test stays deterministic.
5. Import the module in `pimm/datasets/transform/__init__.py`.
6. Verify on a synthetic numpy `data_dict`.
