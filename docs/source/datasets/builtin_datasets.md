# Built-in datasets

pimm ships a handful of dataset classes covering the common detector formats.
They all register with `@DATASETS.register_module()` and are built from config
through the `DATASETS` registry. This page is a tour of what each one does; for
PILArNet specifically see {doc}`pilarnet`.

## The registry

The dataset registry lives in `pimm/datasets/builder.py`:

```python
DATASETS = Registry("datasets")
```

Configs name a class by its registered `type`:

```python
dict(type="PILArNetH5Dataset", split="train", transform=transform)
```

:::{important}
**Registration is by import side effect.** `pimm/datasets/__init__.py` imports
the dataset modules so their decorators run. A class defined but never imported
there is *not* buildable from config — `DATASETS.build(...)` will not find it.
This is the most common "why can't it find my dataset?" gotcha, and it also
matters on resume (which reloads the dumped config). See
{doc}`../getting_started/concepts`.
:::

### Auto-imported classes

These are imported by `pimm/datasets/__init__.py` and always available:

```{list-table}
:header-rows: 1
:widths: 26 74

* - `type`
  - Summary
* - `DefaultDataset`
  - generic directory / JSON-split dataset of `.npy` assets.
* - `ConcatDataset`
  - combines child dataset configs and repeats them with `loop`.
* - `PILArNetH5Dataset`
  - direct HDF5 reader for PILArNet-M point clouds ({doc}`pilarnet`).
* - `JAXTPCDataset`
  - multimodal LArTPC dataset with separate seg / resp / corr / labl readers.
* - `LUCiDDataset`
  - Water Cherenkov sensor and segment dataset.
* - `LUCiDEventSSLDataset`
  - pimm-data backed LUCiD event dataset for SSL classification-style
    pretraining.
* - `UBooNEH5Dataset`
  - HDF5 reader for MicroBooNE semantic-segmentation data.
```

### Registered but not auto-imported

These register with the decorator but are **not** imported by
`pimm/datasets/__init__.py`, so a config (or other module) must import them
first before `DATASETS.build(...)` can find them:

- `PandaEventEditorCacheDataset` (`pimm/datasets/panda_event_editor_cache.py`):
  cache-backed supervised dataset for Panda event-editor models.
- `LUCiDRingPanopticDataset` (`pimm/datasets/lucid_ring_panoptic.py`):
  overlap-aware Cherenkov ring panoptic dataset.

:::{note}
A config `__import__` is enough to trigger registration *for that run*, but it
is dropped from the dumped config on resume. For anything you rely on long term,
import it in the package `__init__.py` instead. See the import-side-effect note
above.
:::

## Reader / dataset separation

Two layers do different jobs:

```text
reader (pimm/datasets/readers/)   dataset (pimm/datasets/)
  ─ NOT registry-built              ─ @DATASETS.register_module()
  ─ file discovery + indexing       ─ merges modalities
  ─ lazy HDF5 open after fork       ─ computes/selects labels
  ─ read_event(idx) -> flat dict    ─ chooses standard keys (coord/energy/...)
  ─ no transforms                   ─ attaches metadata, runs Compose(transform)
```

Readers are **not** registry-built and run no transforms. Their shared
convention:

- `__init__`: discover shard files and build an event index.
- `h5py_worker_init`: lazily open HDF5 handles **after** DataLoader worker fork
  (h5py handles are not fork-safe).
- `read_event(idx)`: return a flat `dict[str, np.ndarray]`.
- `__len__`: indexed event count.
- `close`: release file handles.

The dataset class is the boundary where modalities merge, labels are computed,
the standard `coord` / `energy` / `segment` / `instance` keys are chosen, and
metadata is added.

## DefaultDataset

The generic point-cloud dataset for data already arranged as per-sample numpy
assets. It looks under `data_root/split/*`, or treats `data_root/split` as a
JSON list of relative paths when it is a file. Each sample directory holds `.npy`
assets named from `VALID_ASSETS`: `coord`, `color`, `normal`, `strength`,
`segment`, `instance`, `pose`. Missing `segment` / `instance` are filled with
`-1`, coordinates/features are cast to float, then the transform pipeline runs.

## ConcatDataset

Combines several dataset configs:

```python
dict(type="ConcatDataset",
     datasets=[dataset_a_cfg, dataset_b_cfg],
     loop=1)
```

It builds each child with {py:func}`~pimm.datasets.builder.build_dataset` and stores `(dataset_id, index)`
pairs. pimm also provides `MultiDatasetDataloader` (`pimm/datasets/dataloader.py`),
which expects a `ConcatDataset`, uses each child's `loop` as its sampling ratio,
batches from one child at a time, and uses the first child to set epoch length.

## JAXTPCDataset

The multimodal LArTPC dataset for JAXTPC production output. It expects co-indexed
modality shards under a single root:

```text
dataset_root/
  seg/   sim_seg_0000.h5
  resp/  sim_resp_0000.h5
  corr/  sim_corr_0000.h5
  labl/  sim_labl_0000.h5
```

A config selects `modalities` from `("seg", "resp", "corr", "labl")`:

```python
dict(
    type="JAXTPCDataset",
    data_root="/path/to/jaxtpc",
    split="train",
    dataset_name="sim",
    modalities=("seg", "labl"),
    label_key="particle",
    min_deposits=1024,
    transform=transform,
)
```

The dataset owns fusion and decides which spatial source becomes `coord` /
`energy`:

- With `seg`: `coord` is 3D deposits `(N, 3)`.
- No `seg` but `corr` + `labl`: `coord` is labeled 2D correspondence entries
  `(E, 2)`.
- No `seg` or labeled `corr` but `resp`: `coord` is merged 2D wire response
  `(M, 2)`.

Secondary modalities survive under namespaced keys (`resp_coord`, `corr_coord`,
`corr_segment`, raw plane keys like `plane.volume_0_U.wire`, labl lookups like
`labl_v0_particle`, ...). `label_key` selects which labl field becomes `segment`
(`particle`, `cluster`, or `interaction`).

:::{warning}
`labl` alone is only a track-id → label lookup table. For 2D labels you need
`corr` joined with `labl`; `resp` + `labl` cannot produce labels without
correspondence. The JAXTPC readers are split by modality (`JAXTPCSegReader`,
`JAXTPCRespReader`, `JAXTPCCorrReader`, `JAXTPCLablReader`).
:::

:::{note}
JAXTPC resp/corr coordinates can be **2D** `(wire, time)`. Choose transforms
that support 2D, or configure axes accordingly — many geometric transforms
assume 3D. See {doc}`transforms`.
:::

## LUCiDDataset

Reads Water Cherenkov segment and PMT sensor HDF5 files. Accepts
`modalities=("seg",)`, `("sensor",)`, or both. Discovery supports both
`{dataset_name}_seg_*.h5` / `{dataset_name}_sensor_*.h5` and
`segment_events_*.h5` / `sensor_events_*.h5` naming.

```python
dict(
    type="LUCiDDataset",
    data_root="/path/to/lucid",
    split="",
    dataset_name="wc",
    modalities=("sensor",),
    output_mode="response",
    include_labels=True,
    transform=transform,
)
```

Sensor output depends on `output_mode`:

```{list-table}
:header-rows: 1
:widths: 20 80

* - `output_mode`
  - Result
* - `response`
  - one point per PMT. `coord` is PMT position `(N, 3)` when available, else
    sensor index `(N, 1)`. `energy` is total PE, `time` is first-hit time.
* - `labels`
  - sparse per-particle sensor hits; `coord`, `energy`, `time`, `segment`,
    `instance` are point-aligned per hit.
* - `separate`
  - keeps raw reader keys (`pmt_pe`, `pmt_t`, `pp_*`), prefixing 3D segment keys
    as `seg3d.*` when `seg` is also loaded.
```

For segment data, `coord` is each track segment's midpoint, with `energy`,
`time`, `track_ids`, `pdg`, and `parent_ids`. Readers: `LUCiDSegReader`
(CSR track segments → midpoint point clouds) and `LUCiDSensorReader` (PMT event
response + optional sparse per-particle hit decomposition).

:::{note}
`LUCiDDataset` constructs `LUCiDSensorReader` without exposing its
`pmt_positions` / `pmt_positions_file` arguments. To get 3D PMT coordinates,
store positions in the HDF5 file (`config/pmt_positions`); otherwise expect the
1D sensor-index coordinate.
:::

## LUCiDEventSSLDataset

Adapts the external `pimm_data` LUCiD reader for Sonata-style SSL. It reads
multiple LUCiD config directories, builds a deterministic per-config holdout, and
emits event-level labels:

```python
dict(
    type="LUCiDEventSSLDataset",
    data_root=data_root,
    configs=lucid_configs_train,
    split="train",
    dataset_name="wc",
    holdout_events_per_config=512,
    min_points=128,
    aggregate_sensor_hits=True,
    time_aggregation="earliest",
    transform=transform,
)
```

Each `configs` entry is a string or a mapping with `name`, `label`,
`label_name`, `data_root`, and `pimm_split` (the per-config `data_root`
overrides the base). Output keys: `coord`, `energy`, `time`, `sensor_idx`,
`event_label`, `config_id`, `name`, `split`. With `aggregate_sensor_hits=True`,
repeated hits collapse by sensor id (energy summed; time aggregated by
`earliest`, `pe_weighted`, `mean`, or `first`). This dataset does not support
fragmented `test_mode`.

:::{warning}
LUCiD config holdout is seeded by **list position**. A config shared between a
train list and a probe/eval list must sit at the **same index** in both, or
train/heldout leakage results.
:::

## UBooNEH5Dataset

HDF5 reader for MicroBooNE semantic-segmentation data, following the same
reader-then-dataset pattern: it discovers shards, opens handles lazily after
worker fork, and emits the standard `coord` / `energy` / `segment` keys for the
semantic-segmentation task.

## Data roots and modality meaning

Prefer passing `data_root` explicitly in configs for shared runs — only some
constructors infer roots (PILArNet from `PILARNET_DATA_ROOT_*`, see
{doc}`pilarnet`). `JAXTPCDataset` and `LUCiDDataset` require `data_root`;
`LUCiDEventSSLDataset` requires a base `data_root` that each `configs` entry may
override.

The meaning of `coord` and `energy` is dataset- and modality-specific:

```{list-table}
:header-rows: 1
:widths: 40 60

* - Source
  - `coord` / `energy`
* - PILArNet
  - 3D detector voxel coordinate; point energy.
* - JAXTPC seg
  - 3D deposit coordinate.
* - JAXTPC resp/corr (no seg)
  - 2D `(wire, time)` point.
* - LUCiD sensor `response`
  - PMT position if available, else 1D sensor index; `energy` is PE.
* - LUCiD segment
  - track-segment midpoint.
```

Do not assume every dataset has RGB color, normals, a 3D coordinate, or a single
label scheme. Use {py:class}`~pimm.datasets.transform.base.Collect` and task-specific {py:class}`~pimm.datasets.transform.base.Copy`/mapping transforms to make
the final model contract explicit.

## See also

- {doc}`pilarnet` — the primary dataset in depth.
- {doc}`transforms` — turning these keys into a model batch.
- {doc}`bring_your_own` — write and register a new dataset.
