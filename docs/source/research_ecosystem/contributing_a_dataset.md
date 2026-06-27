# Contributing a dataset

Adding a dataset to pimm is **one registered class**. It does three things: index
your events, read one event into a flat numpy dict, and run a transform pipeline
that turns that dict into the packed batch a model consumes. No separate reader,
no base class to satisfy beyond `torch.utils.data.Dataset` — the same shape as
the built-in {py:class}`~pimm.datasets.pilarnet.PILArNetH5Dataset`.

:::{seealso}
For a complete, runnable example that takes a custom dataset all the way to a
trained semantic-segmentation model, see the tutorial
{doc}`../tutorials/byo_dataset_semseg`. This page is the reference walkthrough;
that one is the end-to-end story.
:::

## 1. The contract

Register a `torch.utils.data.Dataset` in the `DATASETS` registry. These are the
only methods that matter:

```{list-table}
:header-rows: 1
:widths: 36 14 50

* - Method
  - Required?
  - What it does / returns
* - `__init__(self, data_root, split, transform=None, …)`
  - yes
  - Index your events up front (cheap — no open file handles), and build the
    pipeline with `self.transform = Compose(transform)`.
* - `get_data(self, idx)`
  - by convention
  - One **raw** event as a flat `dict[str, np.ndarray]`: `coord` `(N, 3)`, your
    feature(s) e.g. `energy` `(N, 1)`, `segment` `(N,)`, plus `name` / `split`
    metadata. Raw numpy, before any transform.
* - `__getitem__(self, idx)`
  - yes
  - `return self.transform(self.get_data(idx))` — the transformed sample the
    loader collates into a packed batch.
* - `__len__(self)`
  - yes
  - Number of samples (times `loop` if you support epoch-stretching).
```

Two invariants keep everything downstream working: build the pipeline with
`Compose(transform)` in `__init__` (pimm hands you the raw **list of dicts**,
never a prebuilt `Compose`), and return **raw numpy** from `get_data` —
{py:class}`~pimm.datasets.transform.base.ToTensor` converts late in the pipeline
and the geometry transforms need numpy. Name your primary spatial array `coord`
so transforms like {py:class}`~pimm.datasets.transform.spatial.NormalizeCoord` /
{py:class}`~pimm.datasets.transform.spatial.GridSample` find it.

## 2. Write the dataset class

This is the whole thing — discovery + indexing in `__init__`, one raw event in
`get_data`, transforms applied in `__getitem__`. The HDF5 specifics are an
example; swap them for your format.

```python
# pimm/datasets/my_dataset.py
import glob, os
import h5py
import numpy as np
from torch.utils.data import Dataset

from .builder import DATASETS
from .transform import Compose


@DATASETS.register_module()
class MyDataset(Dataset):
    def __init__(self, data_root, split="train", transform=None, loop=1, max_len=None):
        super().__init__()
        self.split = split
        self.loop = loop
        self.max_len = max_len
        self.transform = Compose(transform)          # list of dicts -> pipeline

        # Index every event up front as (file, row). Cheap: no open handles yet
        # (h5py handles aren't fork-safe, so we open them lazily per worker below).
        self.files = sorted(glob.glob(os.path.join(data_root, f"*{split}*.h5")))
        self.index = []
        for fi, path in enumerate(self.files):
            with h5py.File(path, "r") as f:
                self.index += [(fi, i) for i in range(f["coord"].shape[0])]
        self._handles = {}

    def get_data(self, idx):
        fi, row = self.index[idx % len(self.index)]
        if fi not in self._handles:                  # open lazily, once per worker
            self._handles[fi] = h5py.File(self.files[fi], "r", swmr=True)
        f = self._handles[fi]
        return {
            "coord":   f["coord"][row].astype(np.float32),                 # (N, 3)
            "energy":  f["energy"][row].astype(np.float32).reshape(-1, 1), # (N, 1)
            "segment": f["label"][row].astype(np.int64),                   # (N,)
            "name":    f"{self.split}_{idx}",
            "split":   self.split,
        }

    def __getitem__(self, idx):
        return self.transform(self.get_data(idx))    # transforms run here, like PILArNet

    def __len__(self):
        n = len(self.index) * self.loop
        return min(n, self.max_len) if self.max_len else n
```

:::{important}
**Return raw numpy in `get_data`.** Conversion to tensors is
{py:class}`~pimm.datasets.transform.base.ToTensor`'s job, late in the pipeline;
returning tensors early breaks the numpy-based geometry transforms. Always attach
`name` / `split` metadata — evaluators and hooks expect them.
:::

:::{tip}
If your whole dataset fits a different layout (one file per event, `.npz`, a
ROOT export, in-memory arrays…), only `__init__` and `get_data` change — index
whatever you have, and return the same flat numpy dict. The lazy-open dance above
is only needed because h5py handles don't survive the DataLoader worker fork; a
one-file-per-event reader can just `np.load` in `get_data`.
:::

## 3. Register it

Import your dataset from `pimm/datasets/__init__.py` so the
`@DATASETS.register_module()` decorator runs:

```python
# pimm/datasets/__init__.py
from .my_dataset import MyDataset   # noqa: F401
```

## 4. Handle new point-aligned keys

If `get_data` returns extra arrays of length `N` that must follow point
subsampling ({py:class}`~pimm.datasets.transform.spatial.GridSample`, etc.), register them in `index_valid_keys` **before**
the first subsampling transform, using the `Update` transform:

```python
dict(type="Update", keys_dict={
    "index_valid_keys": ["coord", "energy", "segment", "my_extra_point_key"],
}),
```

Keys not in `index_valid_keys` keep their original length and silently mismatch
`coord` after subsampling. See {doc}`../datasets/transforms` for the default list.

## 5. What the collate function expects

You never set a `collate_fn` on your dataset — the trainer picks it when it builds
the loader (the default is the point-cloud
{py:func}`~pimm.datasets.utils.collate_fn`). Your only job is to emit per-sample
dicts it can pack. A few rules follow from how it works:

- **Tensors are concatenated, not stacked.** Point-aligned arrays from every
  sample are joined into one ragged `(N_total, …)` tensor — there is no batch
  dimension. Events stay separable only through `offset`.
- **`offset` is mandatory and cumulative.** Any key containing `offset` is treated
  as per-sample point counts and accumulated across the batch.
  {py:class}`~pimm.datasets.transform.base.Collect` stamps the single-sample
  `offset = [N]`; after collation `offset[-1] == N_total`. Without it the model
  can't tell where one event ends.
- **`_`-prefixed keys are dropped.** Use a leading underscore for scratch or
  un-collatable metadata you don't want batched.
- **Strings become a list** (one entry per sample) — fine for `name`. Everything
  else falls through to `default_collate`, so scalars stack normally.
- **Same width across samples.** Per-point arrays must be tensors of identical
  trailing shape (e.g. `feat` is always `(N, C)`) so `cat` succeeds.
  {py:class}`~pimm.datasets.transform.base.ToTensor` does the conversion late in
  the pipeline — keep `get_data` numpy.

If your method genuinely needs a different batching scheme (e.g. flattening a
per-sample *list* of queries, like
{py:func}`~pimm.datasets.utils.inseg_collate_fn`), that's a **trainer** concern,
not a dataset one — subclass the trainer's `build_train_loader` rather than
changing the dataset. See {doc}`contributing_a_model`.

## 6. Stay resume-safe (stateful loading)

The training loader is a torchdata `StatefulDataLoader` driven by
{py:class}`~pimm.datasets.stateful.StatefulRandomSampler`, so a run can resume
mid-epoch without reshuffling or replaying consumed samples. That puts a few
requirements on your class:

- **Be map-style and deterministic in `idx`.** Provide integer `__getitem__` +
  `__len__` and make index → event a fixed mapping. The sampler owns ordering and
  hands you positions, so never keep an internal "next sample" cursor or shuffle
  inside the dataset. Random *augmentation* in transforms is fine — the loader
  snapshots worker RNG.
- **Keep `len(dataset)` stable.** The sampler records the dataset length and
  **refuses to resume** if it changes (`Sampler state length … does not match
  dataset length`). Don't let `__len__` depend on wall-clock or files appearing
  mid-run; a fixed `loop` multiplier set in `__init__` is fine.
- **Stay rank-agnostic.** The sampler handles distributed sharding and padding
  (`num_replicas`/`rank`); don't shard the data yourself.
- **Open non-fork-safe handles lazily.** h5py files, DB connections, etc. must be
  opened inside `get_data`/`__getitem__` and cached per process (the lazy-open
  pattern in §2). Opening them in `__init__` corrupts state across forked
  workers and breaks both throughput and resume.

Stick to a plain map-style dataset and you're on the supported path — exotic
loaders that can't snapshot their state force checkpoints to epoch boundaries
only.

## 7. Write a config

Add a config with a transform pipeline ending in `ToTensor` + {py:class}`~pimm.datasets.transform.base.Collect`, and a
`data` block pointing at your class:

```python
transform = [
    dict(type="NormalizeCoord", center=[384.0, 384.0, 384.0], scale=665.1076),
    dict(type="LogTransform", min_val=0.13, max_val=20.0),
    dict(type="GridSample", grid_size=0.001, mode="train", return_grid_coord=True),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "segment"),
         feat_keys=("coord", "energy")),
]

data = dict(
    num_classes=5,
    ignore_index=-1,
    names=["shower", "track", "michel", "delta", "led"],
    train=dict(type="MyDataset", data_root="/path/to/data", split="train",
               transform=transform),
    val=dict(type="MyDataset", data_root="/path/to/data", split="val",
             transform=transform),
)
```

Loader knobs (`batch_size`, `num_worker_per_gpu`, `seed`, `mix_prob`) go at the
**top level** of the config, not in `data.train`. See
{doc}`../configuration/index`.

## 8. Verify a sample and a batch

Because the transform runs inside `__getitem__`, `ds[0]` is already a transformed
sample. Confirm it — and a collated batch — have the shapes your model expects:

```python
from pimm.datasets import build_dataset
from pimm.datasets.utils import collate_fn

ds = build_dataset(dict(
    type="MyDataset", data_root="/path/to/data", split="train",
    transform=transform,
))

# (a) one transformed sample
sample = ds[0]
print(sorted(sample.keys()))        # coord, grid_coord, segment, offset, feat, ...
print(sample["coord"].shape, sample["feat"].shape)

# (b) a small collated batch
batch = collate_fn([ds[i] for i in range(3)])
print(batch["coord"].shape)         # (N_total, 3)
print(batch["feat"].shape)          # (N_total, C)  -- C == in_channels
print(batch["offset"])              # cumulative, last == N_total
assert batch["offset"][-1].item() == batch["coord"].shape[0]
```

If `feat` has the channel count your backbone's `in_channels` expects and
`offset` is cumulative with `offset[-1] == N_total`, the dataset speaks the
packed contract correctly. See {doc}`../datasets/data_format`.

## Checklist

1. Subclass `torch.utils.data.Dataset`, register with `@DATASETS.register_module()`.
2. `__init__`: index events (no open handles) and build `self.transform = Compose(transform)`.
3. `get_data` returns a flat numpy dict with `coord` / `energy` / `segment` and `name` / `split`.
4. `__getitem__` returns `self.transform(self.get_data(idx))`; `__len__` returns a **stable** count.
5. Map-style + deterministic in `idx`, rank-agnostic, non-fork-safe handles opened lazily (resume-safe).
6. Import the dataset module in `pimm/datasets/__init__.py`.
7. New point-aligned keys added to `index_valid_keys`.
8. Config pipeline ends in `ToTensor` + `Collect` (stamps `offset`); loader knobs at top level.
9. Verify `dataset[0].keys()` and a collated batch's shapes (`offset[-1] == N_total`).

## See also

- {doc}`contributing_a_transform` — the augmentation/preprocessing steps your pipeline runs.
- {doc}`../tutorials/byo_dataset_semseg` — the full end-to-end tutorial.
- {doc}`../datasets/data_format` — the batch contract you are targeting.
- {doc}`../datasets/transforms` — `index_valid_keys`, `Collect`, and the pipeline.
- {doc}`All registered datasets <../api/registry/datasets>` — `PILArNetH5Dataset` and friends to crib from.
