# Training a semantic segmentation model

**Goal.** You have your own detector point clouds - 3D hits with an energy (or
charge) feature and a per-point class label - and you want to train a
[Point Transformer V3](https://arxiv.org/abs/2312.10035) backbone to predict
those classes. By the end you'll have a registered dataset, a config, and a
trained, evaluated model.

This tutorial assumes you've read
{doc}`../getting_started/concepts` and have pimm {doc}`installed
<../getting_started/installation>`.

```text
your raw events ─▶ Reader ─▶ Dataset ─▶ Compose(transform) ─▶ packed batch
                                                                   │
                          DefaultSegmentorV2(PT-v3m2 backbone) ◀───┘
                                  │  per-point seg_logits
                                  ▼  CE + Lovász loss
                              SemSegEvaluator (mIoU)
```

## 0. What the model needs from your data

A semantic-segmentation batch must arrive as a packed dict
(see {doc}`../datasets/data_format`):

| Key | Shape | Meaning |
|-----|-------|---------|
| `coord` | `(N, 3)` | point coordinates (post-normalization) |
| `feat` | `(N, C)` | backbone input features, e.g. `xyz + energy` → `C=4` |
| `offset` | `(B,)` | cumulative event boundaries |
| `segment` | `(N,)` | per-point integer class label (`-1` = ignore) |

Everything below exists to produce that batch from *your* files.

## 1. Write a Dataset class

A pimm dataset is **one** ordinary `torch.utils.data.Dataset` registered in the
`DATASETS` registry - no separate reader, no base class to satisfy. Its whole job
is to turn an index into a flat numpy dict (`coord`, your feature, `segment`, …)
and run a transform pipeline that produces the packed batch from section 0. The
interface is small - these are the only methods that matter:

```{list-table}
:header-rows: 1
:widths: 38 14 48

* - Method
  - Required?
  - What it does / returns
* - `__init__(self, split, data_root, transform=None, …)`
  - yes
  - Index your events (cheap, no open file handles) and build the pipeline with
    `self.transform = Compose(transform)`.
* - `get_data(self, idx)`
  - by convention
  - One **raw** event as a flat `dict[str, np.ndarray]`: `coord` `(N, 3)`, your
    feature(s) e.g. `energy` `(N, 1)`, `segment` `(N,)`, plus `name` / `split`
    metadata strings. Raw numpy, before any transform.
* - `__getitem__(self, idx)`
  - yes
  - `return self.transform(self.get_data(idx))` - the transformed sample the
    loader collates into a packed batch.
* - `__len__(self)`
  - yes
  - Number of samples (multiply by `loop` if you support epoch-stretching).
```

Two rules keep the rest working: build the pipeline with `Compose(transform)` in
`__init__` (pimm passes you the raw **list of dicts**, never a prebuilt
`Compose`), and return **raw numpy** from `get_data` - `ToTensor` converts late
in the pipeline and the geometry transforms need numpy. Name your primary
spatial array `coord` so transforms like `NormalizeCoord` / `GridSample` find it.

Here's the whole class - discovery + indexing in `__init__`, one raw event in
`get_data`, transforms applied in `__getitem__` (the same shape as the built-in
`PILArNetH5Dataset`):

```python
# pimm/datasets/my_tpc.py
import glob

import h5py
import numpy as np
from torch.utils.data import Dataset

from pimm.datasets.builder import DATASETS
from pimm.datasets.transform import Compose


@DATASETS.register_module()
class MyTPCDataset(Dataset):
    def __init__(self, split, data_root, transform=None, min_points=128,
                 max_len=None, loop=1):
        self.split = split
        self.data_root = data_root
        self.transform = Compose(transform)        # list of dicts -> pipeline
        self.min_points = min_points
        self.loop = loop
        # Discover shards and build an event index. Open files lazily per worker.
        self.files = sorted(glob.glob(f"{data_root}/*{split}*.h5"))
        self._index = self._build_index()
        self._handles = {}
        self.max_len = max_len

    def _build_index(self):
        index = []
        for fi, path in enumerate(self.files):
            with h5py.File(path, "r") as f:
                n = f["n_points"][:]                # per-event point counts
            index += [(fi, ei) for ei, c in enumerate(n) if c >= self.min_points]
        return index

    def _file(self, fi):                            # lazy open after worker fork
        if fi not in self._handles:
            self._handles[fi] = h5py.File(self.files[fi], "r")
        return self._handles[fi]

    def get_data(self, idx):
        fi, ei = self._index[idx % len(self._index)]
        f = self._file(fi)
        return {
            "coord":  f["coord"][ei].astype(np.float32),    # (N, 3)
            "energy": f["energy"][ei].astype(np.float32).reshape(-1, 1),
            "segment": f["label"][ei].astype(np.int64),     # (N,)
            "name": f"{self.files[fi]}:{ei}",
            "split": self.split,
        }

    def __getitem__(self, idx):
        return self.transform(self.get_data(idx))

    def __len__(self):
        n = len(self._index) * self.loop
        return min(n, self.max_len) if self.max_len else n
```

:::{important}
**Register where the package imports it.** Add `from . import my_tpc` to
`pimm/datasets/__init__.py` so the `@DATASETS.register_module()` decorator runs.
:::

## 2. Re-derive your transforms

**Do not reuse PILArNet's normalization numbers.** Your detector has its own
coordinate range and energy scale. Derive three things from a sample of your
data:

```{list-table}
:header-rows: 1
:widths: 26 74

* - Transform
  - How to set it
* - `NormalizeCoord(center, scale)`
  - Pick `center` ≈ the detector center and `scale` so coords land in ~`[-1, 1]`.
    `NormalizeCoord` computes `(coord - center) / scale`.
* - `LogTransform(min_val, max_val)`
  - Energy is heavy-tailed; log-compress it. Set `min_val` to your energy
    threshold (so it maps near 0) and `max_val` above your typical maximum.
* - `GridSample(grid_size)`
  - Set `grid_size` to the **minimum normalized point spacing** in your data.
    Too coarse drops real hits; too fine wastes memory. Verify the kept fraction.
```

A quick way to sanity-check `grid_size` and normalization on real events:

```python
from pimm.datasets.builder import build_dataset
import numpy as np

ds = build_dataset(dict(type="MyTPCDataset", split="train",
                        data_root="/path/to/data", transform=[]))
ev = ds.get_data(0)
print("coord range", ev["coord"].min(0), ev["coord"].max(0))
print("energy pct", np.percentile(ev["energy"], [1, 50, 99]))
```

## 3. Write the config

Copy `configs/panda/semseg/semseg-pt-v3m2-pilarnet-ft-5cls-fft.py`
as a starting point and change the dataset + transforms. The model side
({py:class}`~pimm.models.default.DefaultSegmentorV2` wrapping a `PT-v3m2` backbone) stays largely the same.

```python
# configs/mytpc/semseg-ptv3.py
_base_ = ["../_base_/default_runtime.py"]

# --- runtime ---
batch_size = 16          # global; split across GPUs automatically
num_worker = 12
enable_amp = True
amp_dtype = "bfloat16"
matmul_precision = "high"
seed = 0
evaluate = True
use_wandb = True
wandb_project = "MyTPC-SemSeg"

num_classes = 3
names = ["track", "shower", "other"]

# --- model: PTv3 backbone + linear segmentation head ---
model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=1232,
    backbone=dict(
        type="PT-v3m2",
        in_channels=4,                       # xyz + energy
        order=("hilbert", "hilbert-trans", "z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 9, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(256, 256, 256, 256, 256),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 96, 192, 384),
        dec_num_head=(4, 6, 12, 24),
        dec_patch_size=(256, 256, 256, 256),
        mlp_ratio=4, qkv_bias=True, drop_path=0.3,
        shuffle_orders=True, pre_norm=True,
        enable_flash=True,                   # set False if no FlashAttention
        enc_mode=True,                       # encoder-only
        freeze_encoder=False,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0 / 20.0,
             ignore_index=-1),
    ],
    mlp_head=False,
    freeze_backbone=False,
)

# --- optimizer / scheduler ---
epoch = 20
eval_epoch = 20
optimizer = dict(type="AdamW", lr=1.5e-3, weight_decay=0.01)
scheduler = dict(type="OneCycleLR", max_lr=1.5e-3, pct_start=0.05,
                 anneal_strategy="cos", div_factor=10.0, final_div_factor=1000.0)

# --- data + transforms (TUNE these for your detector) ---
grid_size = 0.01
DET_CENTER = [0.0, 0.0, 0.0]
DET_SCALE = 1000.0

transform = [
    dict(type="NormalizeCoord", center=DET_CENTER, scale=DET_SCALE),
    dict(type="LogTransform", min_val=0.13, max_val=20.0),
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv",
         mode="train", return_grid_coord=True),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
    dict(type="RandomFlip", p=0.5),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "segment"),
         feat_keys=("coord", "energy")),
]
test_transform = [
    dict(type="NormalizeCoord", center=DET_CENTER, scale=DET_SCALE),
    dict(type="LogTransform", min_val=0.13, max_val=20.0),
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv",
         mode="train", return_grid_coord=True),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "segment"),
         feat_keys=("coord", "energy")),
]

data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=names,
    train=dict(type="MyTPCDataset", split="train", data_root="/path/to/data",
               transform=transform, min_points=128),
    val=dict(type="MyTPCDataset", split="val", data_root="/path/to/data",
             transform=test_transform, min_points=128, max_len=1000),
    test=dict(type="MyTPCDataset", split="test", data_root="/path/to/data",
              transform=test_transform, min_points=128),
)

# --- hooks: evaluator BEFORE saver so "best" sees the metric ---
hooks = [
    dict(type="WandbNamer", keys=("model.type", "data.train.max_len", "seed")),
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator", every_n_steps=1000, write_cls_iou=True),
    dict(type="CheckpointSaver", save_freq=None, evaluator_every_n_steps=1000),
    dict(type="FinalEvaluator", test_last=False),
]

test = dict(type="SemSegTester", verbose=True)
```

:::{tip}
`in_channels=4` matches `feat_keys=("coord", "energy")` (3 + 1). If you add or
drop a feature, update **both**. The backbone's first layer width is fixed at
this number.
:::

## 4. Quick check

Tiny limits, no W&B, no workers - verifies the whole path before you commit GPUs:

```bash
pimm launch --train.config mytpc/semseg-ptv3 --run.name quickcheck \
  -- epoch=1 data.train.max_len=64 data.val.max_len=32 \
     batch_size=4 num_worker=0 use_wandb=False
```

If it runs an epoch and writes `exp/mytpc/quickcheck/`, your dataset + config are
wired correctly.

## 5. Train

```bash
# single GPU
pimm launch --train.config mytpc/semseg-ptv3 --run.name semseg-ptv3-v1 \
  --resources.nproc-per-node 1

# four GPUs on one node - no Slurm needed (global batch_size splits to 4/GPU)
pimm launch --train.config mytpc/semseg-ptv3 --resources.nproc-per-node 4
```

On a cluster, submit through a site profile.
Site profiles are per-cluster; see {doc}`../hpc/sites` for defining one for your cluster.

```bash
pimm submit --site mycluster --resources.nnodes 1 --resources.nproc-per-node 4 \
  --resources.time 04:00:00 --train.config mytpc/semseg-ptv3
```

Watch it with `tail -f exp/mytpc/semseg-ptv3-v1/train.log` or in W&B
(see {doc}`../hpc/monitoring`). The {py:class}`~pimm.engines.hooks.eval.semantic_segmentation.SemSegEvaluator` logs mIoU/mAcc/F1 and marks
`model_best.pth` on the best mIoU.

## 6. Resume

If a run stops, continue it exactly (RNG, dataloader position, step, optimizer
all restored - even mid-epoch):

```bash
pimm launch --train.config mytpc/semseg-ptv3 --run.name semseg-ptv3-v1 \
  --train.resume
```

See {doc}`../checkpoints/resuming`.

## 7. Evaluate the trained model

```bash
sh scripts/test.sh -c mytpc/semseg-ptv3 -n semseg-ptv3-v1 -w model_best
```

This runs `pimm/test.py` against the experiment's code snapshot and the
`SemSegTester` from your config. See {doc}`../evaluation/index`.

To run inference in your own script, load with {py:func}`~pimm.from_pretrained` and reproduce
the **val** transform - see {doc}`../research_ecosystem/using_trained_models` and
{doc}`../datasets/transforms`.

## 8. (Optional) Fine-tune from a pretrained backbone

Self-supervised pretraining usually buys a large head start. If you have a
Sonata/Panda backbone checkpoint, load its weights into your backbone and remap
the keys (`student.backbone.*` → `backbone.*`):

```python
# add to cfg.hooks (replaces the plain CheckpointLoader)
dict(
    type="CheckpointLoader",
    keywords="module.student.backbone",
    replacement="module.backbone",
),
```

```bash
pimm launch --train.config mytpc/semseg-ptv3 --run.name semseg-ptv3-pt \
  --train.weight hf://<your-org>/sonata-pilarnet-L/model_best.pth
```

Because `--train.resume` is **not** set, only weights load - optimizer and
schedule start fresh, which is what you want for a new task. A remap that matches
zero parameters raises, so a silent random-init can't happen. See
{doc}`../checkpoints/saving_and_loading`.

## Where to go next

- {doc}`panda_detector` - graduate from semantic to panoptic (per-instance) with
  the Panda Detector.
- {doc}`../configuration/index` - make reusable config variants.
- {doc}`../datasets/transforms` - the full transform catalog.
