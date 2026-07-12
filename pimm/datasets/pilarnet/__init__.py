"""PILArNet-M dataset package.

Public surface is unchanged from the former ``pimm.datasets.pilarnet`` module -
importing this package registers the datasets with ``DATASETS`` and re-exports
the shared decode helpers.

Submodules:
    - ``decode``: :func:`decode_event`, :func:`map_instance_ids` (shared by both readers)
    - ``h5``: :class:`PILArNetH5Dataset`
    - ``parquet``: :class:`PILArNetParquetDataset`, :class:`PILArNetParquetIterableDataset`
"""

from .decode import DEFAULT_LABEL_PRIORITY, decode_event, map_instance_ids
from .h5 import PILArNetH5Dataset
from .parquet import (
    PILArNetParquetDataset,
    PILArNetParquetIterableDataset,
    resolve_parquet_data_files,
)

__all__ = [
    "DEFAULT_LABEL_PRIORITY",
    "decode_event",
    "map_instance_ids",
    "PILArNetH5Dataset",
    "PILArNetParquetDataset",
    "PILArNetParquetIterableDataset",
    "resolve_parquet_data_files",
]
