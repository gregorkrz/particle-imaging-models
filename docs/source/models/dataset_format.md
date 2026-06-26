# Feeding a model the right data

The single most common reason a loaded model gives garbage predictions is a
**preprocessing mismatch**: the input wasn't normalized, log-transformed, or
gridded the way it was during training. A pimm model is only meaningful on data
that went through the *same* transform pipeline and arrives as the *same* packed
batch. This page shows how to reproduce both.

:::{important}
Coordinate normalization and the energy {py:class}`~pimm.datasets.transform.color.LogTransform` are **part of the model's
contract**, not optional cosmetics. For example, the PoLAr-MAE checkpoints
require `LogTransform(min_val=0.13)` (the energy threshold) — using `0.01`
produces near-random results. Always reuse the transform from the run's saved
config.
:::

## Where the right transform lives

Every run writes its resolved config next to the checkpoints. The transform you
need is in there:

```python
from pimm.utils.config import Config

cfg = Config.fromfile("exp/panda/semseg/my-run/config.py")
val_transform = cfg.data.val.transform   # the exact list of transform dicts used at val time
```

If you exported the model, the same information is in
`training_config.json` under `["data"]`. **Reuse it** rather than re-deriving
the magic numbers.

## Build a packed batch by hand

Models consume a packed batch: 2D `(N, C)` tensors with a cumulative `offset`
vector (see {doc}`../datasets/packed_format`). The cleanest way to produce one is
to run the dataset's own transform pipeline and collate function:

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

# 2. A raw event as a flat dict of numpy arrays — same keys the dataset produces.
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
`feat_keys`, match them — the backbone's `in_channels` is fixed at training time.

:::{tip}
`collate_fn` **concatenates** the per-event tensors and builds the cumulative
`offset`; pass a list of several samples to batch multiple events. Keys whose
names start with `_` are dropped during collation.
:::

## …or just use the dataset

Often the least error-prone path is to build the dataset from the saved config
and let it do everything:

```python
from pimm.datasets.builder import build_dataset
from pimm.datasets.utils import collate_fn

dataset = build_dataset(cfg.data.val)     # applies the val transform internally
sample = dataset[0]
batch = collate_fn([dataset[i] for i in range(4)])
```

Then run inference exactly as in {doc}`index`.

## Coordinate & energy gotchas

```{list-table}
:header-rows: 1
:widths: 36 64

* - Pitfall
  - Fix
* - Wrong `LogTransform` `min_val`
  - Use the value from the run's config (e.g. `0.13` for PoLAr-MAE = the energy
    threshold). The original library mutates `emin` to the threshold internally.
* - Skipped `NormalizeCoord`
  - `NormalizeCoord(scale=X)` computes `(coord - center) / scale`. Many models
    normalize to roughly `[-1, 1]^3`; the constant `768 * sqrt(3) / 2 ≈ 665.1`
    is common for PILArNet.
* - Wrong `feat_keys`
  - `feat` must have the same channel count and order as training, or the
    backbone's first layer is fed nonsense.
* - Forgetting `grid_coord`
  - Sparse backbones expect the gridded coordinate from `GridSample`. Keep
    `return_grid_coord=True` and `Collect` it.
* - `low_energy_scatters`
  - Some evaluations need `remove_low_energy_scatters=True` to match the trained
    class scheme. Mirror the dataset setting from the config.
```

## Checklist

- Read the transform from the run's `config.py` / `training_config.json` — don't
  invent numbers.
- Match `feat_keys` (and therefore `in_channels`) exactly.
- Keep `grid_coord` if the model uses a sparse backbone.
- Collate into a packed batch (`coord`, `feat`, `offset`, plus any extras).
- Move tensors to the model's device before calling `model(batch)`.

## See also

- {doc}`index` — loading the model itself.
- {doc}`../datasets/transforms` — the transform pipeline in depth.
- {doc}`../datasets/packed_format` — the packed-batch contract.
