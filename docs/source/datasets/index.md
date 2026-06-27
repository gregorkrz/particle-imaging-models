# Datasets

pimm datasets are **map-style PyTorch datasets that return flat dictionaries of
numpy arrays**. A config names a dataset class and its transforms; the dataset
loads raw arrays and attaches metadata; a `Compose(transform)` pipeline
augments and projects them; and a collate function packs the per-sample dicts
into the ragged `(N, C)` + `offset` batch the models expect.

The package is intentionally thin, with three clear responsibilities:

```text
reader   ──▶  HDF5 file discovery, lazy handle opening, raw modality decoding
dataset  ──▶  event-level fusion, label selection, standard keys + metadata
transform──▶  augmentation, tensor conversion, final key/feature selection
```

If you read one page first, read {doc}`data_format` — the `(N, C)` + `offset`
contract is the single idea that lets the same models run on argon TPCs, water
Cherenkov detectors, and wire-plane data.

- {doc}`Data format <data_format>` — the `(N, C)` + cumulative `offset` contract, and how `collate_fn` builds it.
- {doc}`Transforms <transforms>` — `Compose`, common transforms, `index_valid_keys`, the final `Collect`, and reproducing the pipeline for inference.
- {doc}`PILArNet-M <pilarnet>` — download, revisions v1/v2/v3, env vars, output keys, and event overlay.
- {doc}`All registered datasets <../api/registry/datasets>` — `PILArNetH5Dataset`, `DefaultDataset`, `ConcatDataset`, JAXTPC, LUCiD, UBooNE, and the rest, generated from the `DATASETS` registry.
- {doc}`Core concepts <../getting_started/concepts>` — how datasets fit into the registry / config / trainer picture.

## The common path

```text
config.py  ──▶  data.train / data.val / data.test   (constructor dicts)
           ──▶  build_dataset(cfg)                   (DATASETS registry)
                  └─ dataset loads numpy arrays + attaches name/split
                  └─ Compose(transform) augments and projects
           ──▶  collate_fn  ──▶  packed batch  { coord, feat, offset, ... }
```

1. A Python config defines `data.train`, `data.val`, and optionally
   `data.test` as constructor dictionaries with a `type` key.
2. `build_dataset(cfg)` builds the configured class through the `DATASETS`
   registry.
3. The dataset loads numpy arrays, attaches metadata such as `name` and
   `split`, and runs its `Compose(transform)` pipeline.
4. The loader collates transformed samples into packed tensors, usually with
   `coord`, `feat`, and `offset` keys.

## Config shape at a glance

Dataset configuration lives under the top-level `data` dictionary:

```python
data = dict(
    num_classes=5,
    ignore_index=-1,
    names=["shower", "track", "michel", "delta", "led"],
    train=dict(
        type="PILArNetH5Dataset",
        revision="v2",
        split="train",
        transform=transform,
        min_points=1024,
        loop=1,
    ),
    val=dict(type="PILArNetH5Dataset", revision="v2", split="val",
             transform=val_transform, max_len=1000),
    test=dict(type="PILArNetH5Dataset", revision="v2", split="test",
              transform=test_transform),
)
```

:::{important}
**Loader settings are not dataset constructor arguments.** `batch_size`,
`batch_size_per_gpu`, `num_worker_per_gpu`, `seed`, and `mix_prob` live at the
**top level** of the config, not inside `data.train`. The default trainer wires
`data.train` to a `StatefulDataLoader` + `StatefulRandomSampler` for exact
mid-epoch resume; validation and testing use plain `DataLoader`s. See
{doc}`../configuration/index`.
:::

## Next

- {doc}`data_format` — the batch contract every model speaks.
- {doc}`transforms` — augmentation and the final `Collect` projection.
- {doc}`pilarnet` — the primary LArTPC dataset.
- {doc}`All registered datasets <../api/registry/datasets>` — every registered class, generated from the registry.

```{toctree}
:hidden:

data_format
transforms
pilarnet
```
