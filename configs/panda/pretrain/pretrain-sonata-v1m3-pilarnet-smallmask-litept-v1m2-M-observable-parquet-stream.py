"""Streaming (IterableDataset) variant of the parquet SONATA/LitePT pretrain.

Same as ``...-parquet.py`` but the TRAIN split streams via
:class:`PILArNetParquetIterableDataset` (sequential shard reads + a shuffle
buffer -- the access pattern NVMe/Lustre reward) instead of map-style random
access. Validation stays map-style. Everything else (model, transforms,
optimizer, mask schedule) is inherited.

Epoch length: a stream has no ``__len__``, so the engine sizes ``iters_per_epoch``
from the dataset's ``num_samples()`` (``num_samples // world_size //
batch_size_per_gpu``, floored for DDP lockstep) unless you set ``iters_per_epoch``
below. ``num_samples()`` needs a local parquet root (stage via
``scripts/pilarnet/stage_lscratch.sh``; resolves through ``PILARNET_PARQUET_ROOT_V2``).

Note: ``test_mode`` and event overlay are not supported by the streaming reader
(neither is used here). ``loop``/``max_len`` from the base config are tolerated
(``loop`` warns and is ignored; ``max_len`` caps the derived epoch length).
"""

_base_ = ["./pretrain-sonata-v1m3-pilarnet-smallmask-litept-v1m2-M-observable-parquet.py"]

# Swap only the train reader to streaming; deep-merge keeps revision/split/
# transform/energy_threshold/min_points from the base.
data = dict(
    train=dict(
        type="PILArNetParquetIterableDataset",
        shuffle=True,
        shuffle_buffer_size=2000,
    ),
)

# iters_per_epoch = 30000  # optional: override the num_samples-derived default.
