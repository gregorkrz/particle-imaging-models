"""Parquet-reader variant of the constant-mask SONATA/LitePT PILArNet pretrain.

Identical to ``pretrain-sonata-v1m3-pilarnet-smallmask-litept-v1m2-M-observable``
except the dataset is read from our zstd parquet
(:class:`PILArNetParquetDataset`) instead of HDF5. Everything else -- model,
transforms, optimizer, mask schedule, loader settings -- is inherited unchanged.

Point the reader at a parquet root one of two ways:
  - set ``PILARNET_PARQUET_ROOT_V2`` (e.g. a ``/lscratch`` staging dir; see
    ``scripts/pilarnet/stage_lscratch.sh``), which the reader picks up
    automatically, or
  - ``--options data.train.data_root=/path/to/parquet/v2``.

The parquet is produced offline from the HDF5 shards (h5 -> zstd converter).

Note: the parquet reader supports event overlay (same ``overlay_*`` kwargs as
the H5 reader) but not ``test_mode``; neither is used here (train/val pretrain).
"""

_base_ = ["./pretrain-sonata-v1m3-pilarnet-smallmask-litept-v1m2-M-observable.py"]

# Swap only the reader type; deep-merge keeps revision/split/transform/
# energy_threshold/min_points/max_len/loop from the base data dict.
data = dict(
    train=dict(type="PILArNetParquetDataset"),
    val=dict(type="PILArNetParquetDataset"),
)
