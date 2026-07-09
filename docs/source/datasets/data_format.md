# Data format

Detector events are **variable length** - one neutrino interaction might be 100
hits, the next 10,000. The usual way to batch variable-length events for
stochastic gradient descent is to **pad** them into a dense 3D tensor of shape
`(B, N, C)` - batch size `B`, a fixed maximum `N` points per event, and `C`
per-point features. When event sizes vary widely that wastes a lot of memory on
meaningless zeros.

So pimm batches point clouds **packed**: every batched quantity is 2D `(N, C)`
instead of 3D `(B, N, C)` - the concatenation of all events into one flat tensor
- with an `offset` tensor marking where each event ends. This is the
[Compressed Sparse Row (CSR)](<https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)>)
layout, the same one graph neural networks use because it operates naturally on
variable-length objects.

```text
event 0: 100 pts ┐
event 1: 150 pts ├─▶  coord:  (330, 3)   feat: (330, C)
event 2:  80 pts ┘     offset: [100, 250, 330]   # cumulative
```

<p align="center">
    <img alt="packed offsets" class="only-light" src="https://raw.githubusercontent.com/pointcept/assets/main/pointcept/offset.png" width="480">
    <img alt="packed offsets" class="only-dark" src="https://raw.githubusercontent.com/pointcept/assets/main/pointcept/offset_dark.png" width="480">
</p>

This is the format that lets the *same* model run on argon TPCs, water
Cherenkov PMT arrays, and 2D wire-plane data - only the meaning of `coord` and
`feat` changes, never the batch layout. See {doc}`../getting_started/concepts`
for where this sits in the wider picture.

:::{warning}
**Packed memory is spiky.** Because the batch size *in points* depends on which
events land together, VRAM use isn't constant and the allocator
[fragments](<https://en.wikipedia.org/wiki/Fragmentation_(computing)>) - a batch
with more hits than usual can OOM you partway through a run. Log VRAM over
training with
{py:class}`~pimm.engines.hooks.resources.ResourceUtilizationLogger`.
:::

## The offset vector

`offset` is the **cumulative sum** of the per-event point counts. For events of
length 100, 150, and 80:

```text
lengths = [100, 150,  80]
offset  = [100, 250, 330]      # cumsum; offset[-1] == total_points == N
```

`offset[i]` is the **end index** (exclusive) of event `i` in the packed
tensors, so event `i` occupies rows `offset[i-1] : offset[i]` (with an implicit
`offset[-1] = 0` before the first event). `offset` has length `batch_size`, and
`offset[-1]` equals the total number of points `N`.

:::{tip}
`offset` is conceptually identical to PyG's `batch` vector - one labels event
boundaries by cumulative count, the other labels each point with its event id.
pimm's {py:class}`~pimm.models.utils.structure.Point` structure
(`pimm/models/utils/structure.py`) derives one from the other, and also carries
the sparse-CNN bookkeeping (serialized order, batch indices) that backbones like
PT-v3 need - so backbones can use either representation.
:::

## A model-facing batch

After collation a supervised semantic-segmentation batch typically looks like:

```python
{
    "coord":   Tensor[total_points, D],   # D = 2 or 3 spatial dims
    "feat":    Tensor[total_points, C],   # features fed to the backbone
    "offset":  Tensor[batch_size],        # cumulative event boundaries
    "segment": Tensor[total_points],      # per-point labels (when supervised)
    "name":    list[str],                 # per-event identifiers
}
```

The full specification - dtypes, and when each key is present:

```{list-table}
:header-rows: 1
:widths: 16 20 14 50

* - Key
  - Shape
  - Dtype
  - Present / notes
* - `coord`
  - `(N, D)`, D = 2 or 3
  - `float32`
  - **always** - point coordinates after the transform pipeline
* - `feat`
  - `(N, C)`
  - `float32`
  - **always** - concat of `Collect(feat_keys=...)`; `C` must equal the backbone's `in_channels`
* - `offset`
  - `(B,)`
  - `int64`
  - **always** - cumulative per-event point counts; `offset[-1] == N`
* - `grid_coord`
  - `(N, 3)`
  - `int64`
  - sparse backbones (PT-v3, SpUNet, Mink) - integer voxel indices from `GridSample(return_grid_coord=True)`
* - `segment`
  - `(N,)`
  - `int64`
  - supervised - per-point class id (`ignore_index` for unlabeled). Detector configs read a task-specific name (e.g. `segment_pid`)
* - `instance`
  - `(N,)`
  - `int64`
  - instance / panoptic - per-point instance id (task-specific name, e.g. `instance_particle`)
* - `name`
  - `(B,)` list
  - `str`
  - **always** - per-event identifiers (a `list`, not a tensor)
```

Dtypes follow {py:class}`~pimm.datasets.transform.base.ToTensor` (integer arrays → `int64`,
floating → `float32`); `Collect` casts `feat` to float and records `offset` as
`int64`.

Note what is *and is not* batched:

- **Point-aligned tensors** (`coord`, `feat`, `segment`, `instance`, ...) are
  concatenated along dim 0, so they all share the same `total_points` length.
- **`offset`** is the only per-event structural tensor.
- **Metadata** like `name` becomes a plain Python `list[str]`.

## How `collate_fn` builds it

The collate functions live in `pimm/datasets/utils.py`. {py:func}`~pimm.datasets.utils.collate_fn`
walks each sample recursively and applies a few rules:

```{list-table}
:header-rows: 1
:widths: 32 68

* - Leaf type
  - Collation rule
* - tensor
  - **concatenated** along dim 0 (not stacked)
* - string
  - gathered into a Python `list`
* - sequence
  - transposed and collated element-wise; a length tensor is appended and
    converted to cumulative offsets
* - mapping
  - collated key by key, **excluding keys whose names start with `_`**
* - key containing `offset`
  - treated as per-sample offsets and re-accumulated into cumulative offsets
    for the whole batch
```

Two consequences are worth internalizing:

- **Concatenate, not stack.** There is no batch dimension to index. A model that needs per-event slices reads them out of `offset`.
- **Per-sample offsets become global offsets.** Each sample arrives with its
  own single-element offset (built by {doc}`Collect <transforms>` from
  `coord.shape[0]`). Collation stitches these into the running cumulative
  vector - you never compute the cumulative sum yourself.

:::{important}
**Metadata keys starting with `_` are dropped by collation.** Use a leading
underscore (e.g. `_event_path`, `_raw_meta`) for anything you want available on
the per-sample dict for debugging but *not* in the batched tensors. Anything
without the underscore must be collatable.
:::

### Variants

```{list-table}
:header-rows: 1
:widths: 26 74

* - Function
  - Purpose
* - {py:func}`~pimm.datasets.utils.collate_fn`
  - the base recursive collator described above.
* - {py:func}`~pimm.datasets.utils.point_collate_fn`
  - wraps `collate_fn` and adds `mix_prob`, which can merge paired point clouds
    (adjusting instance ids and shrinking the `offset` vector) for mix-style
    augmentation.
```

`mix_prob` is supplied from the **top-level** config, not the dataset - it is a
loader-side knob. The default training loader uses
{py:class}`~pimm.datasets.stateful.StatefulRandomSampler` + `StatefulDataLoader`
so a mid-epoch checkpoint restores the exact sampler position; validation and
testing use plain `DataLoader`s.

## Verifying the format

The safest check for any dataset is to collate a tiny batch and inspect shapes:

```python
from pimm.datasets.utils import collate_fn

samples = [dataset[i] for i in range(3)]
batch = collate_fn(samples)

print(batch["coord"].shape)    # (N_total, 3)
print(batch["feat"].shape)     # (N_total, C)
print(batch["offset"])         # cumulative, last entry == N_total
assert batch["offset"][-1].item() == batch["coord"].shape[0]
```
