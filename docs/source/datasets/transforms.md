# Transforms

A transform pipeline turns the **flat numpy dict** a dataset produces into the
**packed tensor batch** a model consumes. Transforms own augmentation, label
remapping, tensor conversion, and the final key/feature selection — datasets
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
pipeline from the dicts internally (and the dumped config must stay JSON-like so
it round-trips through resume). Hand it the raw `[dict(type=...), ...]` list.
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
  - Compresses `energy` with a log scale (clipped to `[min_val, max_val]`). For
    PILArNet the **`min_val` must equal the energy threshold** (`0.13`); using
    the wrong floor produces garbage features.
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
up as `segment` and `instance` by the time the model sees the batch — copy or
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

{py:class}`~pimm.datasets.transform.base.Collect` is the projection that produces the model contract. Given:

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
   {doc}`packed_format`).
3. **Builds feature tensors** from any `*_keys` keyword: each such argument is
   `torch.cat`'d along dim 1, so `feat_keys=("coord", "energy")` yields
   `feat = concat([coord, energy], dim=1)` — here a 4-channel `[x, y, z, E]`
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

Every transform registered in `TRANSFORMS` — drop any of them into a `transform`
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

:::{tip}
`PDGToSemantic` carries the `pid_6cls` PDG→class map used by the Panda detector
(`{22:0, 11:1, 13:2, 211:3, 2212:4}`, everything else → `5` / "led"); see
{doc}`../models/dataset_format`.
:::

## See also

- {doc}`packed_format` — what `Collect` + collation produce.
- {doc}`pilarnet` — the output keys these pipelines consume.
- {doc}`bring_your_own` — adding transforms for new point-aligned keys.
