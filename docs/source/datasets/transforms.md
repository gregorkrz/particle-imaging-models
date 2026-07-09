# Transforms

A transform pipeline turns the **flat numpy dict** a dataset produces into the
**packed tensor batch** a model consumes. Transforms own augmentation, label
remapping, tensor conversion, and the final key/feature selection - datasets
stay focused on loading and fusing raw arrays.

## `transform` is a list of dicts

The single most important convention: `transform` is passed as a **list of
config dictionaries**, never a pre-built {py:class}`~pimm.datasets.transform.base.Compose`.

```python
transform = [
    dict(type="NormalizeCoord", center=[384.0, 384.0, 384.0], scale=665.1076),
    dict(type="LogTransform", min_val=0.13, max_val=20.0),
    dict(type="GridSample", grid_size=0.001, mode="train", return_grid_coord=True),
    dict(type="Copy", keys_dict={"segment_motif": "segment"}),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "segment"),
         feat_keys=("coord", "energy")),
]
```

`Compose` (in `pimm/datasets/transform/base.py`) builds each entry through the
`TRANSFORMS` registry and applies them in order. Transforms registered with
`@TRANSFORMS.register_module()` receive and return the same `data_dict` object
by convention (a few return a reduced dict).

:::{warning}
Do **not** pass a pre-instantiated `Compose` in a config. pimm rebuilds the
pipeline from the dicts internally. Hand it the raw `[dict(type=...), ...]` list.
:::

## A worked pipeline

The order below is the standard semantic-segmentation shape used across the
PILArNet configs. Each step is explained underneath.

```python
grid_size = 0.001  # in normalized coordinate units
transform = [
    dict(type="NormalizeCoord", center=[384.0, 384.0, 384.0],
         scale=768.0 * 3**0.5 / 2),
    dict(type="LogTransform", min_val=0.13, max_val=20.0),
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv",
         mode="train", return_grid_coord=True),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1, 1], axis="x", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1, 1], axis="y", center=[0, 0, 0], p=0.8),
    dict(type="RandomFlip", p=0.5),
    dict(type="Copy", keys_dict={"segment_motif": "segment"}),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "segment"),
         feat_keys=("coord", "energy")),
]
```

```{list-table}
:header-rows: 1
:widths: 24 76

* - Step
  - What it does
* - `NormalizeCoord`
  - Recenters and rescales `coord`: `coord = (coord - center) / scale`. The
    constant `768 * sqrt(3) / 2 ≈ 665.1076` is the half-diagonal of the
    PILArNet `768³` volume, mapping the detector into roughly `[-1, 1]³`.
* - `LogTransform`
  - Compresses `energy` with a log scale (clipped to `[min_val, max_val]`).
* - `GridSample`
  - Voxelizes onto a hash grid of size `grid_size` (in *normalized* units) and
    deduplicates points per voxel. `mode="train"` samples one point per voxel;
    `return_grid_coord=True` also emits integer `grid_coord` for sparse convs.
* - `RandomRotate` / `RandomFlip`
  - Geometric augmentation around the origin. These act on `coord` (and rotate
    any registered vector-valued point keys).
* - `Copy`
  - Duplicates a key. The canonical use is selecting which label becomes the
    conventional `segment` key (`segment_motif -> segment`).
* - `ToTensor`
  - Converts numpy/scalars to tensors. Place it **near the end**, after the
    numpy-based geometry and filtering steps.
* - `Collect`
  - The final projection: keeps `keys`, builds `offset` from `coord`, and
    concatenates `feat` from `feat_keys`.
```

## `coord` is the canonical spatial key

Geometric transforms ({py:class}`~pimm.datasets.transform.spatial.NormalizeCoord`, {py:class}`~pimm.datasets.transform.spatial.GridSample`, {py:class}`~pimm.datasets.transform.spatial.RandomRotate`,
{py:class}`~pimm.datasets.transform.spatial.RandomFlip`, ...) operate on `coord`. Whatever your dataset calls its primary
spatial array, it must be named `coord` for these to find it. Labels usually end
up as `segment` and `instance` by the time the model sees the batch - copy or
remap dataset-specific labels into those names before `Collect`.

:::{note}
Some transforms assume **3D** coordinates. For 2D inputs (e.g. JAXTPC wire
response or correspondence points with `coord` of shape `(N, 2)`), pick
transforms that support 2D or configure axes carefully.
:::

## Keeping point-aligned arrays length `N`: `index_valid_keys`

Subsampling transforms (such as `GridSample` in train mode) reduce the point
count. They call `index_operator`, which applies the selected-point index **only
to keys listed in `index_valid_keys`**. Any point-aligned array *not* in that
list keeps its original length and silently desynchronizes from `coord`.

`index_valid_keys` defaults (in `pimm/datasets/transform/common.py`) already
include the standard keys:

```text
coord, color, normal, strength, segment, instance, energy, local_shape,
segment_motif, segment_pid, instance_particle, instance_interaction,
momentum, vertex, segment_interaction, ...   (+ is_primary when revision == "v3")
```

If you add a **new point-aligned key**, register it before any subsampling
transform with the `Update` transform:

```python
dict(type="Update", keys_dict={
    "index_valid_keys": [
        "coord", "energy", "segment",
        "my_new_point_key",          # <-- now follows the selected points
    ],
}),
```

:::{important}
Add new point-aligned keys to `index_valid_keys` **before** the first
subsampling transform (typically `GridSample`). A key introduced after
subsampling, or never registered, will not be indexed and will mismatch `N`.
:::

## The final `Collect`

{py:class}`~pimm.datasets.transform.base.Collect` is the projection that produces the model input format. Given:

```python
dict(type="Collect",
     keys=("coord", "grid_coord", "segment"),
     feat_keys=("coord", "energy"))
```

it does three things:

1. **Copies the requested `keys`** through unchanged.
2. **Builds offsets** from `offset_keys_dict` (default `dict(offset="coord")`),
   stamping a single-element per-sample offset = `coord.shape[0]`. Collation
   later re-accumulates these into the cumulative batch offset (see
   {doc}`data_format`).
3. **Builds feature tensors** from any `*_keys` keyword: each such argument is
   `torch.cat`'d along dim 1, so `feat_keys=("coord", "energy")` yields
   `feat = concat([coord, energy], dim=1)` - here a 4-channel `[x, y, z, E]`
   feature. You can define multiple feature groups (e.g. `feat_keys=...` and a
   second `*_keys` argument) and each becomes its own concatenated tensor named
   after the prefix.

:::{tip}
`feat_keys` is how the backbone's `in_channels` is determined. `("coord",
"energy")` over 3D coords gives `in_channels=4`. Change the feature set here, not
in the model.
:::

## Metadata and the `_` convention

Keys starting with `_` are **skipped by collation**. Use them for per-sample
debugging metadata you do not want in the batched tensors. Anything you `Collect`
without a leading underscore must be a collatable tensor, string, or sequence.

## Transform catalog

Every transform registered in `TRANSFORMS` - drop any of them into a `transform`
list by `type`. Full constructor signatures are in the
{doc}`API reference <../api/index>`.

```{list-table}
:header-rows: 1
:widths: 26 74

* - Group
  - Transforms
* - **Projection & utility**
  - `Collect`, `Copy`, `ToTensor`, `Update`
* - **Coordinate geometry**
  - `NormalizeCoord`, `CenterShift`, `PositiveShift`, `PointClip`, `RandomRotate`,
    `RandomRotateTargetAngle`, `RandomFlip`, `RandomScale`, `RandomShift`,
    `RandomJitter`, `ClipGaussianJitter`, `MultiplicativeRandomJitter`,
    `ElasticDistortion`
* - **Sampling & cropping**
  - `GridSample`, `SphereCrop`, `CropBoundary`, `HardExampleCrop`, `RandomDrop`,
    `RandomDropout`, `ShufflePoint`
* - **Features / energy / color**
  - `LogTransform`, `MomentumTransform`, `RelativeLogNormalize`, `EnergyJitter`,
    `EnergeticTranslation`, `NormalizeColor`, `ChromaticAutoContrast`,
    `ChromaticJitter`, `ChromaticTranslation`, `HueSaturationTranslation`,
    `RandomColorDrop`, `RandomColorGrayScale`, `RandomColorJitter`
* - **Labels & instances**
  - `PDGToSemantic`, `InstanceParser`, `ComputeAnchors`, `LocalCovarianceFeatures`
* - **Multi-view & SSL**
  - `MultiViewGenerator`, `ContrastiveViewsGenerator`,
    `MixedScaleGeometryMultiViewGenerator`, `HierarchicalMaskGenerator`,
    `HMAECollate`
* - **Control flow**
  - `ConditionalRandomTransform`, `SetRandomValue`
```

## Reproducing the pipeline at inference

The single most common reason a loaded model gives garbage predictions is a
**preprocessing mismatch**: the input wasn't normalized, log-transformed, or
gridded the way it was at training time. A model is only meaningful on data that
went through the *same* transform pipeline and arrives as the *same* packed batch
({doc}`data_format`).

:::{important}
Coordinate normalization and the energy {py:class}`~pimm.datasets.transform.color.LogTransform` are **part of the
model's input format**, not optional cosmetics. The PoLAr-MAE checkpoints, for
example, require `LogTransform(min_val=0.13)` (the energy threshold) - using
`0.01` produces near-random results. Always reuse the transform from the run's
saved config.
:::

### Reuse the transform from the run's config

Every run writes its resolved config next to the checkpoints, and an export
carries the same information under `config.json` (`["data"]`). Read it instead of
re-deriving the magic numbers:

```python
from pimm.utils.config import Config

cfg = Config.fromfile("exp/panda/semseg/my-run/config.py")
val_transform = cfg.data.val.transform   # the exact list of transform dicts used at val time
```

### Build a packed batch by hand

Models consume a packed batch: 2D `(N, C)` tensors with a cumulative `offset`
vector ({doc}`data_format`). The cleanest way to produce one is to run the
dataset's own transform pipeline and collate function:

```python
import numpy as np
import torch
from pimm.datasets.transform import Compose
from pimm.datasets.utils import collate_fn

# 1. The transform pipeline from the run's config (list of dicts).
transform = Compose([
    dict(type="NormalizeCoord", center=[384.0, 384.0, 384.0], scale=768.0 * 3**0.5 / 2),
    dict(type="LogTransform", min_val=0.13, max_val=20.0),
    dict(type="GridSample", grid_size=0.001, hash_type="fnv",
         mode="train", return_grid_coord=True),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord"), feat_keys=("coord", "energy")),
])

# 2. A raw event as a flat dict of numpy arrays - same keys the dataset produces.
event = {
    "coord":  coords_np.astype(np.float32),    # (N, 3) raw detector coordinates
    "energy": energy_np.astype(np.float32),    # (N, 1) raw energy
    "name":   "evt-0",
}

# 3. Transform, then collate into a packed batch (a list with one sample here).
sample = transform(event)
batch = collate_fn([sample])
# -> {"coord": (N,3), "grid_coord": (N,3), "feat": (N,4), "offset": tensor([N]), ...}
```

`feat_keys=("coord", "energy")` is what makes `feat` 4-dimensional
(`in_channels=4` in most configs: xyz + energy). If your model used different
`feat_keys`, match them - the backbone's `in_channels` is fixed at training time.

### Building the batch from the dataset

Often the least error-prone path is to build the dataset from the saved config
and let it apply the transform internally:

```python
from pimm.datasets.builder import build_dataset
from pimm.datasets.utils import collate_fn

dataset = build_dataset(cfg.data.val)     # applies the val transform internally
batch = collate_fn([dataset[i] for i in range(4)])
```

Then run inference as in {doc}`../research_ecosystem/using_trained_models`.

### Coordinate & energy gotchas

```{list-table}
:header-rows: 1
:widths: 36 64

* - Pitfall
  - Fix
* - Wrong `LogTransform` `min_val`
  - Use the value from the run's config (e.g. `0.13` for PoLAr-MAE = the energy
    threshold).
* - Skipped `NormalizeCoord`
  - `NormalizeCoord(scale=X)` computes `(coord - center) / scale`. Many models
    normalize to roughly `[-1, 1]^3`; the constant `768 * sqrt(3) / 2 ≈ 665.1`
    is common for PILArNet.
* - Wrong `feat_keys`
  - `feat` must have the same channel count and order as training, or the
    backbone's first layer is fed nonsense.
* - Forgetting `grid_coord`
  - Sparse backbones require the gridded coordinate from `GridSample`. Keep
    `return_grid_coord=True` and `Collect` it.
* - `low_energy_scatters`
  - Some evaluations need `remove_low_energy_scatters=True` to match the trained
    class scheme. Mirror the dataset setting from the config.
```

Checklist: read the transform from the run's `config.py` / `config.json` (don't
invent numbers); match `feat_keys` (and therefore `in_channels`) exactly; keep
`grid_coord` if the model uses a sparse backbone; collate into a packed batch;
and move tensors to the model's device before calling `model(batch)`.
