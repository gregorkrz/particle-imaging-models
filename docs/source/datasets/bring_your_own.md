# Bring your own dataset

This page is a practical, numbered walkthrough for adding a new dataset to pimm.
The pattern is always the same: a **reader** owns raw file I/O, a **dataset**
owns fusion and label selection, and a **transform** pipeline owns augmentation
and the final model contract.

:::{seealso}
For a complete, runnable example that takes a custom dataset all the way to a
trained semantic-segmentation model, see the tutorial
{doc}`../tutorials/byo_dataset_semseg`. This page is the reference walkthrough;
that one is the end-to-end story.
:::

## 0. Decide how much you need to write

```text
already per-sample .npy assets?  ──▶  just use DefaultDataset, write a config
raw HDF5 / multimodal?           ──▶  write a reader + a dataset class (below)
```

If your data is already arranged as per-sample `coord.npy`, `segment.npy`, etc.,
you may not need a new class at all — point `DefaultDataset` at it (see
{doc}`builtin_datasets`). Otherwise, continue.

## 1. Write a reader

Readers live in `pimm/datasets/readers/`, are **not** registry-built, and run no
transforms. They own file discovery, indexing, lazy handle opening, raw
decoding, and cleanup. h5py handles are not fork-safe, so open them lazily after
the DataLoader worker fork.

```python
# pimm/datasets/readers/my_reader.py
import glob, os
import h5py
import numpy as np


class MyH5Reader:
    def __init__(self, data_root, split):
        # Discover shards and build an event index up front (cheap; no open handles).
        self.files = sorted(glob.glob(os.path.join(data_root, f"*{split}/*.h5")))
        self._handles = None
        self.index = []  # list of (file_idx, event_idx_within_file)
        for fi, path in enumerate(self.files):
            with h5py.File(path, "r") as f:
                n = f["coord"].shape[0]      # however your file is laid out
            self.index.extend((fi, ei) for ei in range(n))

    def h5py_worker_init(self):
        # Open handles lazily, once per worker, after fork.
        if self._handles is None:
            self._handles = [h5py.File(p, "r") for p in self.files]

    def read_event(self, idx):
        self.h5py_worker_init()
        fi, ei = self.index[idx]
        f = self._handles[fi]
        return {
            "coord":  np.asarray(f["coord"][ei], dtype=np.float32),   # (N, 3)
            "energy": np.asarray(f["energy"][ei], dtype=np.float32),  # (N, 1)
            "segment": np.asarray(f["segment"][ei], dtype=np.int64),  # (N,)
        }

    def __len__(self):
        return len(self.index)

    def close(self):
        if self._handles is not None:
            for h in self._handles:
                h.close()
            self._handles = None
```

`read_event(idx)` must return a **flat `dict[str, np.ndarray]`**. Keep all
physics here; do not augment or convert to tensors.

## 2. Write a dataset class

The dataset inherits `torch.utils.data.Dataset`, registers with
`@DATASETS.register_module()`, and builds its transform pipeline with {py:class}`~pimm.datasets.transform.base.Compose`.

```python
# pimm/datasets/my_dataset.py
import torch
from .builder import DATASETS
from .transform import Compose


@DATASETS.register_module()
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, split="train", data_root=None, transform=None,
                 loop=1, max_len=None):
        super().__init__()
        from .readers.my_reader import MyH5Reader
        self.reader = MyH5Reader(data_root, split)
        self.split = split
        self.loop = loop
        self.max_len = max_len
        # IMPORTANT: pass the raw list of dicts; Compose builds it internally.
        self.transform = Compose(transform)

    def get_data(self, idx):
        # Real index into the reader (loop wraps the logical length).
        data = self.reader.read_event(idx % len(self.reader))
        # Attach standard keys + metadata. coord/energy/segment already present.
        data["name"] = f"{self.split}_{idx}"
        data["split"] = self.split
        return data

    def __getitem__(self, idx):
        return self.transform(self.get_data(idx))

    def __len__(self):
        n = len(self.reader) * self.loop
        return min(n, self.max_len) if self.max_len else n

    def __del__(self):
        try:
            self.reader.close()
        except Exception:
            pass
```

:::{important}
**Return raw numpy in `get_data`.** Conversion to tensors is {py:class}`~pimm.datasets.transform.base.ToTensor`'s job,
late in the pipeline. Returning tensors early breaks numpy-based geometry
transforms. And always attach `name` / `split` metadata — evaluators and hooks
expect them.
:::

:::{tip}
Use `coord` for your primary spatial array and `segment` / `instance` for
labels. Geometric transforms look for `coord` by name; matching the convention
means existing transforms just work. See {doc}`transforms`.
:::

## 3. Register the dataset (import side effect)

The `@DATASETS.register_module()` decorator only runs if the module is imported.
Add it to `pimm/datasets/__init__.py`:

```python
# pimm/datasets/__init__.py
from .my_dataset import MyDataset   # noqa: F401
```

:::{warning}
Importing it only from a config's `__import__` is **not** enough for the long
run: the dumped config replayed on resume drops `__import__`, and
`DATASETS.build(...)` will then fail to find your class. Register it in the
package `__init__.py`. See {doc}`../getting_started/concepts`.
:::

If you added a reader users should import directly, also update
`pimm/datasets/readers/__init__.py`.

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
`coord` after subsampling. See {doc}`transforms` for the default list.

## 5. Write a config

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

## 6. Verify a sample and a batch

Before launching training, confirm the per-sample dict and the collated batch
have the shapes your model expects:

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
packed contract correctly. See {doc}`packed_format`.

## Checklist

1. Reader in `pimm/datasets/readers/` — discovery, lazy open, `read_event`,
   `close`.
2. Dataset subclass registered with `@DATASETS.register_module()`; build
   `self.transform = Compose(transform)`.
3. `get_data` returns a flat numpy dict with `coord` / `energy` / `segment` and
   `name` / `split`.
4. Import the dataset module in `pimm/datasets/__init__.py`.
5. New point-aligned keys added to `index_valid_keys`.
6. Config with `ToTensor` + `Collect`; loader knobs at top level.
7. Verify `dataset[0].keys()` and a collated batch's shapes.

## See also

- {doc}`../tutorials/byo_dataset_semseg` — the full end-to-end tutorial.
- {doc}`packed_format` — the batch contract you are targeting.
- {doc}`transforms` — `index_valid_keys`, `Collect`, and the pipeline.
- {doc}`builtin_datasets` — reference implementations to crib from.
