"""Shared PILArNet event decoding.

``decode_event`` turns one event's raw ``point``/``cluster``/``cluster_extra``
arrays (exactly as stored on disk, from either HDF5 or a parquet row) into the
flat per-point ``data_dict``. Both :class:`PILArNetH5Dataset` and
:class:`PILArNetParquetDataset` call it so the two readers cannot drift.
"""

from typing import Literal

import numpy as np

# priority for voxel deduplication: track (1) > shower (0) > michel (2) > delta (3) > led (4)
DEFAULT_LABEL_PRIORITY = {1: 0, 0: 1, 2: 2, 3: 3, 4: 4}


def decode_event(
    point,
    cluster,
    cluster_extra,
    revision: Literal["v1", "v2", "v3"] = "v2",
    *,
    energy_threshold: float = 0.0,
    remove_low_energy_scatters: bool = False,
    old_pid_mapping: bool = False,
) -> dict:
    """Decode one PILArNet event's raw arrays into a flat ``data_dict``.

    Shared by :class:`PILArNetH5Dataset` (arrays from h5py) and
    :class:`PILArNetParquetDataset` (arrays from a parquet row) so the two
    readers cannot drift. The three inputs are the flat (or already reshaped)
    ``point``/``cluster``/``cluster_extra`` arrays for a single event, exactly as
    stored on disk. ``cluster_extra`` is ignored for ``revision == "v1"`` (pass
    ``None``).

    Returns the per-point ``dict`` (``coord``, ``energy``, ``momentum``,
    ``vertex``, ``segment_motif``, ``segment_pid``, ``instance_particle``,
    ``instance_interaction``, ``segment_interaction``, plus ``is_primary`` for
    v3). Source metadata (``name``/``split``/``revision``) is added by the caller.
    """
    # (x, y, z, e) per point
    data = np.asarray(point).reshape(-1, 8)[:, [0, 1, 2, 3]]

    if revision == "v1":
        # v1: cluster dataset is (-1, 5) without PID, no cluster_extra dataset
        cluster_size, group_id, interaction_id, semantic_id = (
            np.asarray(cluster).reshape(-1, 5)[:, [0, 2, -2, -1]].T
        )
        # v1 doesn't have interaction_id or pid, set defaults
        pid = np.full_like(semantic_id, -1)  # -1
        # v1 doesn't have cluster_extra, set defaults for momentum and vertex
        mom = np.zeros_like(semantic_id, dtype=np.float32)
        vtx_x = np.zeros_like(semantic_id, dtype=np.float32)
        vtx_y = np.zeros_like(semantic_id, dtype=np.float32)
        vtx_z = np.zeros_like(semantic_id, dtype=np.float32)
    elif revision == "v2":
        cluster_size, group_id, interaction_id, semantic_id, pid = (
            np.asarray(cluster).reshape(-1, 6)[:, [0, 2, -3, -2, -1]].T
        )
        mom, vtx_x, vtx_y, vtx_z = (
            np.asarray(cluster_extra).reshape(-1, 5)[:, [1, 2, 3, 4]].T
        )
        pid[pid == -1] = (
            5 if not old_pid_mapping else 6
        )  # -1 (LED) --> 5 (where Kaon is) or 6 (new ID)
    elif revision == "v3":
        cluster_size, group_id, interaction_id, semantic_id, pid = (
            np.asarray(cluster).reshape(-1, 6)[:, [0, 2, -3, -2, -1]].T
        )
        n_clusters = cluster_size.shape[0]
        raw_extra = np.asarray(cluster_extra)
        cluster_extra_arr = (
            raw_extra.reshape(n_clusters, -1)
            if n_clusters > 0
            else np.empty((0, 6), dtype=np.float32)
        )
        if cluster_extra_arr.shape[1] != 6:
            raise ValueError(
                f"Expected v3 cluster_extra width 6, got {cluster_extra_arr.shape[1]}"
            )
        mom, vtx_x, vtx_y, vtx_z, is_primary = cluster_extra_arr[:, [1, 2, 3, 4, 5]].T
        pid[pid == -1] = (
            5 if not old_pid_mapping else 6
        )  # -1 (LED) --> 5 (where Kaon is) or 6 (new ID)
    else:
        raise ValueError(f"Unsupported PILArNet revision: {revision}")

    # np.repeat needs integer counts; parquet may hand back non-int dtypes
    cluster_size = np.asarray(cluster_size).astype(np.int64)

    # Remove low energy scatters if configured
    if remove_low_energy_scatters:
        data = data[cluster_size[0] :]
        semantic_id, group_id, interaction_id, pid, cluster_size = (
            semantic_id[1:],
            group_id[1:],
            interaction_id[1:],
            pid[1:],
            cluster_size[1:],
        )
        mom, vtx_x, vtx_y, vtx_z = mom[1:], vtx_x[1:], vtx_y[1:], vtx_z[1:]
        if revision == "v3":
            is_primary = is_primary[1:]

    # Compute semantic ids for each point
    data_semantic_id = np.repeat(semantic_id, cluster_size)
    data_group_id = np.repeat(group_id, cluster_size)
    data_interaction_id = np.repeat(interaction_id, cluster_size)
    data_pid = np.repeat(pid, cluster_size)
    data_mom = np.repeat(mom, cluster_size)
    data_vtx_x = np.repeat(vtx_x, cluster_size)
    data_vtx_y = np.repeat(vtx_y, cluster_size)
    data_vtx_z = np.repeat(vtx_z, cluster_size)
    if revision == "v3":
        data_is_primary = np.repeat(is_primary, cluster_size)

    # Apply energy threshold if needed
    if energy_threshold > 0:
        threshold_mask = data[:, 3] > energy_threshold
        data = data[threshold_mask]
        data_semantic_id = data_semantic_id[threshold_mask]
        data_group_id = data_group_id[threshold_mask]
        data_interaction_id = data_interaction_id[threshold_mask]
        data_pid = data_pid[threshold_mask]
        data_mom = data_mom[threshold_mask]
        data_vtx_x = data_vtx_x[threshold_mask]
        data_vtx_y = data_vtx_y[threshold_mask]
        data_vtx_z = data_vtx_z[threshold_mask]
        if revision == "v3":
            data_is_primary = data_is_primary[threshold_mask]

    # Prepare return dictionary
    data_dict = {}

    # Get coordinates
    data_dict["coord"] = data[:, :3].astype(np.float32)

    # Process energy (raw)
    energy = data[:, 3].astype(np.float32)
    data_dict["energy"] = energy[:, None]

    # Momentum and vertex labels (v2/v3 only)
    data_dict["momentum"] = data_mom.astype(np.float32)[:, None]
    data_dict["vertex"] = np.stack(
        [data_vtx_x, data_vtx_y, data_vtx_z], axis=1
    ).astype(np.float32)
    if revision == "v3":
        data_dict["is_primary"] = data_is_primary.astype(np.int32)[:, None]

    # Get semantic labels
    data_dict["segment_motif"] = data_semantic_id.astype(np.int32)[:, None]
    data_dict["segment_pid"] = data_pid.astype(np.int32)[:, None]
    # compute both particle- and interaction-level instances
    particle_ids = data_group_id.astype(np.int32)
    interaction_ids = data_interaction_id.astype(np.int32)

    data_dict["instance_particle"] = map_instance_ids(particle_ids)
    data_dict["instance_interaction"] = map_instance_ids(interaction_ids)
    data_dict["segment_interaction"] = (interaction_ids[:, None] != -1).astype(
        np.int32
    )  # 1 if not background, 0 if background

    return data_dict


def map_instance_ids(instance_ids_array):
    """Map instance ids to new ids.

    i.e. instead of having instance ids like [0, 1, 23, 47, 52, 53, 54, 55, 56, 57],
            we want to have instance ids like [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    unique_ids_local = np.unique(instance_ids_array)
    id_mapping_local = {
        old_id: new_id
        for new_id, old_id in enumerate(unique_ids_local[unique_ids_local >= 0])
    }
    return np.array(
        [id_mapping_local.get(id_val, -1) for id_val in instance_ids_array],
        dtype=np.int32,
    )[:, None]
