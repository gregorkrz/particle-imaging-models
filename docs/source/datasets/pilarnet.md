# PILArNet-M

[PILArNet-M](https://huggingface.co/datasets/DeepLearnPhysics/PILArNet-M) is the
primary LArTPC point-cloud dataset in pimm: simulated liquid-argon detector
events as 3D voxelized hits with energy and rich truth labels. The
`PILArNetH5Dataset` class reads the HDF5 shards directly — no preprocessing into
per-sample `.npy` assets is required.

## Download

The 168 GB dataset is hosted on Hugging Face. The downloader is
`scripts/download_pilarnet.py`:

```bash
python scripts/download_pilarnet.py --version v2 --output-dir /path/to/dir
```

```{list-table}
:header-rows: 1
:widths: 28 72

* - Flag
  - Meaning
* - `--version {v1,v2,both}`
  - Which revision(s) to fetch (default `both`).
* - `--output-dir DIR`
  - Base directory; each version lands in `DIR/<revision>`. Defaults to the
    cache under `~/.cache/pimm/pilarnet/`.
* - `--v1-dir` / `--v2-dir`
  - Override the per-version directory explicitly.
```

If `--output-dir` is omitted, data saves to `~/.cache/pimm/pilarnet/<version>`,
which is exactly where the dataset's fallback root inference looks (see below).
After downloading, `cp example.env .env` and set the matching
`PILARNET_DATA_ROOT_*` so the loader finds the data automatically.

:::{note}
Events differ between splits across revisions. A model trained on v1 should be
**evaluated on v1**; do not mix revisions for a single train/eval pair.
:::

## Locating the data

`PILArNetH5Dataset` discovers shards with the glob `*{split}/*.h5` under its
root, filters events by `min_points`, and lazily opens HDF5 handles after the
DataLoader worker fork. The root is resolved in this order:

1. The explicit `data_root` constructor argument (preferred for shared runs).
2. The environment variable matching the revision:
   - `PILARNET_DATA_ROOT_V1`
   - `PILARNET_DATA_ROOT_V2`
3. The fallback `~/.cache/pimm/pilarnet/{revision}` if that directory exists.

:::{important}
Resolution uses ordinary process environment variables, so the **launch path
must export `PILARNET_DATA_ROOT_*` before training starts** (e.g. via `.env` or
the launch YAML). `example.env` documents `_V1` and `_V2`; the code
(`pilarnet.py`) also reads `_V3`.
:::

## Revisions

::::{tab-set}

:::{tab-item} v1
The original PILArNet layout used by the **PoLAr-MAE** paper. Coordinates,
energy, and motif segmentation labels — no PID / momentum / vertex truth. Use it
to reproduce PoLAr-MAE results.
:::

:::{tab-item} v2 (recommended)
Reprocessed PILArNet-M that **adds PID, momentum, and vertex** information per
particle, plus particle- and interaction-level instance ids. The default choice
for new models.
:::

:::{tab-item} v3
Same core layout as v2 **plus interaction-level vertices and primary-particle
labels** (`is_primary`). Needed for tasks that reason about interaction
structure or primary selection.
:::

::::

## Output keys

`get_data(idx)` returns a flat numpy dict. The available keys depend on the
revision:

```{list-table}
:header-rows: 1
:widths: 26 14 60

* - Key
  - Shape
  - Meaning
* - `coord`
  - `(N, 3)`
  - detector voxel coordinates.
* - `energy`
  - `(N, 1)`
  - raw per-point energy deposit.
* - `segment_motif`
  - `(N,)`
  - motif class: `0` shower, `1` track, `2` Michel, `3` delta, `4` low-energy
    deposit (LED).
* - `segment_pid`
  - `(N,)`
  - PID class labels (v2/v3).
* - `instance_particle`
  - `(N,)`
  - remapped particle instance ids.
* - `instance_interaction`
  - `(N,)`
  - remapped interaction instance ids.
* - `segment_interaction`
  - `(N,)`
  - interaction foreground flag.
* - `momentum`
  - `(N, 3)`
  - point-aligned particle momentum (v2/v3).
* - `vertex`
  - `(N, 3)`
  - point-aligned vertex position (v2/v3).
* - `is_primary`
  - `(N,)`
  - primary-particle flag (v3 only).
* - `name`, `split`, `revision`
  - scalar
  - metadata; `revision` also drives `index_valid_keys` (v3 adds `is_primary`).
```

The five motif classes map to the standard config metadata:

```python
data = dict(
    num_classes=5,
    names=["shower", "track", "michel", "delta", "led"],
    ...
)
```

:::{tip}
Most segmentation configs select the motif labels by copying them into the
conventional `segment` key before tensor conversion:

```python
dict(type="Copy", keys_dict={"segment_motif": "segment"})
```

Swap in `segment_pid` for PID tasks. See {doc}`transforms`.
:::

## A minimal config block

```python
data = dict(
    num_classes=5,
    ignore_index=-1,
    names=["shower", "track", "michel", "delta", "led"],
    train=dict(
        type="PILArNetH5Dataset",
        revision="v2",
        split="train",
        # data_root="/path/to/pilarnet-m/v2",   # or rely on env var
        transform=transform,
        energy_threshold=0.13,
        min_points=1024,
        max_len=1_000_000,
        remove_low_energy_scatters=False,
    ),
    val=dict(type="PILArNetH5Dataset", revision="v2", split="val",
             transform=test_transform, min_points=1024, max_len=1000),
    test=dict(type="PILArNetH5Dataset", revision="v2", split="test",
              transform=test_transform, min_points=1024, max_len=1000),
)
```

Common constructor arguments: `revision`, `split`, `data_root`, `transform`,
`min_points`, `max_len`, `loop`, `energy_threshold`, and
`remove_low_energy_scatters`.

:::{warning}
`energy_threshold` and the {py:class}`~pimm.datasets.transform.color.LogTransform` `min_val` are coupled. For PILArNet
the energy floor is `0.13`; the `LogTransform` `min_val` must match it.
`remove_low_energy_scatters=True` is needed for the 4-class motif evaluation
(it drops the LED class). Mismatched values silently degrade results.
:::

## Event overlay

`PILArNetH5Dataset` can synthesize busier events by overlaying multiple raw
events into one, controlled by three constructor arguments:

```{list-table}
:header-rows: 1
:widths: 32 68

* - Argument
  - Effect
* - `overlay_n_events`
  - how many extra events to merge into the base event.
* - `overlay_prob`
  - probability of applying overlay to a given sample.
* - `overlay_allow_repeats`
  - whether the same source event may be drawn more than once.
```

When overlay fires it:

1. **Rotates** each additional event by a random 90-degree increment.
2. **Offsets instance ids** so particles/interactions stay distinct.
3. **Concatenates** point-aligned keys.
4. **Deduplicates** overlapping voxels, resolving collisions by motif priority:
   **track > shower > Michel > delta > low-energy deposit**.

Overlay is an augmentation toggle; leave it off for clean evaluation.

## See also

- {doc}`transforms` — the {py:class}`~pimm.datasets.transform.spatial.NormalizeCoord` / `LogTransform` / {py:class}`~pimm.datasets.transform.spatial.GridSample`
  pipeline these keys flow through.
- {doc}`data_format` — how the per-event dicts become a batch.
- {doc}`All registered datasets <../api/registry/datasets>` — other registered datasets, generated from the registry.
