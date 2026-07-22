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

# sentinel for invalid mass/momentum/vertex in the conversion (matches the
# pxpypz reprocessing scripts)
_INVALID = -1.0


def _leading_cluster_index(cluster_id, group_id, cluster_size):
    """Row index of each group's *leading* cluster, per cluster.

    A particle instance downstream is one ``group_id``, but per-particle truth
    (momentum/mass) is stored per *cluster*. EM showers fragment into several
    clusters under one group (the conversion e- plus secondary e+/brems), each
    carrying DIFFERENT values; collapsing per voxel then lands on whichever
    sub-cluster straddles the middle voxel -- often a low-energy secondary -- so
    the group's truth is taken from the wrong cluster. mlreco sets a group's id
    to the cluster_id of its primary particle, so we reindex every cluster's
    per-particle truth to the group's leading cluster: the one with
    ``cluster_id == group_id``, else (primary left no voxels of its own) the
    largest cluster in the group. Single-cluster (e.g. track) groups are
    unchanged; only fragmented EM groups are corrected.
    """
    cluster_id = np.asarray(cluster_id).astype(np.int64)
    group_id = np.asarray(group_id).astype(np.int64)
    cluster_size = np.asarray(cluster_size).astype(np.int64)
    n = cluster_id.shape[0]
    lead_row: dict[int, int] = {}
    for i in range(n):
        g = int(group_id[i])
        cur = lead_row.get(g)
        if cur is None:
            lead_row[g] = i
            continue
        i_primary = cluster_id[i] == g
        cur_primary = cluster_id[cur] == g
        # prefer the primary (cluster_id == group_id); break ties by size
        if i_primary and not cur_primary:
            lead_row[g] = i
        elif i_primary == cur_primary and cluster_size[i] > cluster_size[cur]:
            lead_row[g] = i
    return np.array([lead_row[int(g)] for g in group_id], dtype=np.int64)


def _true_energy(mom, mass):
    """Per-cluster total energy sqrt(|p|^2 + m^2); ``_INVALID`` where either
    input is the invalid sentinel. ``|p|`` and ``mass`` must both be in GeV."""
    mom = np.asarray(mom, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    valid = (mom >= 0) & (mass >= 0)
    energy = np.sqrt(np.clip(mom * mom + mass * mass, 0.0, None))
    return np.where(valid, energy, _INVALID).astype(np.float32)


def decode_event(
    point,
    cluster,
    cluster_extra,
    revision: Literal["v1", "v2", "v3"] = "v2",
    *,
    cluster_extra_2=None,
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

    Per-particle truth (``momentum``, ``mass``, ``px``/``py``/``pz`` and the
    derived ``true_energy``) is reindexed to each group's leading cluster before
    being broadcast to points -- see :func:`_leading_cluster_index` -- so
    fragmented EM showers no longer inherit a low-energy secondary's momentum.

    Momentum components ``px``/``py``/``pz`` (GeV) are read, when present, from a
    separate ``cluster_extra_2`` dataset (SPLIT layout: width-5
    ``[px, py, pz, parent_pdg, parent_trackid]``) or from the tail of a packed
    ``cluster_extra`` (width 9 -> ``+[px, py, pz]``; width 11 -> ``+parent_pdg,
    parent_id``); when neither is available they are the ``-1`` sentinel.

    Returns the per-point ``dict`` (``coord``, ``energy``, ``momentum``,
    ``mass``, ``px``, ``py``, ``pz``, ``true_energy``, ``vertex``,
    ``segment_motif``, ``segment_pid``, ``instance_particle``,
    ``instance_interaction``, ``segment_interaction``, plus ``is_primary`` for
    v3). Source metadata (``name``/``split``/``revision``) is added by the caller.
    """
    # (x, y, z, e) per point
    data = np.asarray(point).reshape(-1, 8)[:, [0, 1, 2, 3]]

    # per-cluster momentum components + rest mass; sentinel-filled unless the
    # revision/layout provides them (see below).
    def _sentinels(n):
        return (
            np.full(n, _INVALID, dtype=np.float32),  # px
            np.full(n, _INVALID, dtype=np.float32),  # py
            np.full(n, _INVALID, dtype=np.float32),  # pz
        )

    if revision == "v1":
        # v1: cluster dataset is (-1, 5) without PID, no cluster_extra dataset
        cluster_arr = np.asarray(cluster).reshape(-1, 5)
        cluster_size, cluster_id, group_id, interaction_id, semantic_id = (
            cluster_arr[:, [0, 1, 2, -2, -1]].T
        )
        # v1 doesn't have interaction_id or pid, set defaults
        pid = np.full_like(semantic_id, -1)  # -1
        # v1 doesn't have cluster_extra, set defaults for momentum and vertex
        mom = np.zeros_like(semantic_id, dtype=np.float32)
        mass = np.full_like(semantic_id, _INVALID, dtype=np.float32)
        vtx_x = np.zeros_like(semantic_id, dtype=np.float32)
        vtx_y = np.zeros_like(semantic_id, dtype=np.float32)
        vtx_z = np.zeros_like(semantic_id, dtype=np.float32)
        px, py, pz = _sentinels(semantic_id.shape[0])
    elif revision in ("v2", "v3"):
        cluster_arr = np.asarray(cluster).reshape(-1, 6)
        cluster_size, cluster_id, group_id, interaction_id, semantic_id, pid = (
            cluster_arr[:, [0, 1, 2, -3, -2, -1]].T
        )
        n_clusters = cluster_size.shape[0]
        raw_extra = np.asarray(cluster_extra)
        extra = (
            raw_extra.reshape(n_clusters, -1)
            if n_clusters > 0
            else np.empty((0, 6), dtype=np.float32)
        )
        px, py, pz = _sentinels(n_clusters)
        if revision == "v2":
            # cluster_extra: [mass, |p|, vtx_x, vtx_y, vtx_z] (width 5); no px/py/pz
            if extra.shape[1] < 5:
                raise ValueError(
                    f"Expected v2 cluster_extra width >= 5, got {extra.shape[1]}"
                )
            mass, mom, vtx_x, vtx_y, vtx_z = extra[:, [0, 1, 2, 3, 4]].T
        else:  # v3
            # cluster_extra base: [mass, |p|, vtx_x, vtx_y, vtx_z, is_primary].
            # Width 6 = SPLIT layout (px/py/pz live in cluster_extra_2); width
            # 9/11 = PACKED layout with momentum (and lineage) appended.
            if extra.shape[1] not in (6, 9, 11):
                raise ValueError(
                    f"Expected v3 cluster_extra width 6, 9 or 11, got {extra.shape[1]}"
                )
            mass, mom, vtx_x, vtx_y, vtx_z, is_primary = extra[:, [0, 1, 2, 3, 4, 5]].T
            if cluster_extra_2 is not None and extra.shape[1] == 6:
                # SPLIT layout: cluster_extra_2 is (n, 5) ->
                # [px, py, pz, parent_pdg, parent_trackid], row-aligned.
                extra2 = (
                    np.asarray(cluster_extra_2).reshape(n_clusters, -1)
                    if n_clusters > 0
                    else np.empty((0, 5), dtype=np.float32)
                )
                if extra2.shape[1] < 3:
                    raise ValueError(
                        f"Expected cluster_extra_2 width >= 3, got {extra2.shape[1]}"
                    )
                px, py, pz = extra2[:, [0, 1, 2]].T
            elif extra.shape[1] >= 9:
                # PACKED layout: momentum appended at cols 6..8.
                px, py, pz = extra[:, [6, 7, 8]].T
        # cluster_extra[0] is the mlreco PID rest mass in MeV; |p| and (px,py,pz)
        # are in GeV, so convert mass to GeV to keep the 4-momentum consistent.
        # The -1.0 invalid sentinel is preserved (stays negative after scaling).
        mass = np.where(mass == _INVALID, _INVALID, mass / 1.0e3).astype(np.float32)
        pid = pid.copy()
        pid[pid == -1] = (
            5 if not old_pid_mapping else 6
        )  # -1 (LED) --> 5 (where Kaon is) or 6 (new ID)
    else:
        raise ValueError(f"Unsupported PILArNet revision: {revision}")

    # np.repeat needs integer counts; parquet may hand back non-int dtypes
    cluster_size = np.asarray(cluster_size).astype(np.int64)

    # Leading-cluster fix (v2/v3): reindex per-particle truth (momentum/mass and
    # the momentum components) to each group's leading cluster before broadcast.
    # Applied BEFORE remove_low_energy_scatters so the [1:] slice below stays
    # row-consistent. is_primary/vertex are intentionally NOT reindexed.
    if revision in ("v2", "v3") and cluster_size.shape[0] > 0:
        lead_idx = _leading_cluster_index(cluster_id, group_id, cluster_size)
        mom = np.asarray(mom)[lead_idx]
        mass = np.asarray(mass)[lead_idx]
        px = np.asarray(px)[lead_idx]
        py = np.asarray(py)[lead_idx]
        pz = np.asarray(pz)[lead_idx]

    # per-cluster total energy from the (fixed) |p| and mass
    true_energy = _true_energy(mom, mass)

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
        mom, mass, vtx_x, vtx_y, vtx_z = (
            mom[1:], mass[1:], vtx_x[1:], vtx_y[1:], vtx_z[1:]
        )
        px, py, pz, true_energy = px[1:], py[1:], pz[1:], true_energy[1:]
        if revision == "v3":
            is_primary = is_primary[1:]

    # Compute per-point values via repeat
    data_semantic_id = np.repeat(semantic_id, cluster_size)
    data_group_id = np.repeat(group_id, cluster_size)
    data_interaction_id = np.repeat(interaction_id, cluster_size)
    data_pid = np.repeat(pid, cluster_size)
    data_mom = np.repeat(mom, cluster_size)
    data_mass = np.repeat(mass, cluster_size)
    data_px = np.repeat(px, cluster_size)
    data_py = np.repeat(py, cluster_size)
    data_pz = np.repeat(pz, cluster_size)
    data_true_energy = np.repeat(true_energy, cluster_size)
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
        data_mass = data_mass[threshold_mask]
        data_px = data_px[threshold_mask]
        data_py = data_py[threshold_mask]
        data_pz = data_pz[threshold_mask]
        data_true_energy = data_true_energy[threshold_mask]
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

    # Momentum and vertex labels (v2/v3 only; v1 uses zero/sentinel defaults)
    data_dict["momentum"] = data_mom.astype(np.float32)[:, None]
    data_dict["mass"] = data_mass.astype(np.float32)[:, None]
    data_dict["px"] = data_px.astype(np.float32)[:, None]
    data_dict["py"] = data_py.astype(np.float32)[:, None]
    data_dict["pz"] = data_pz.astype(np.float32)[:, None]
    # Packed (N, 3) UNIT momentum-direction vector for regression. Only the
    # *direction* of the momentum is regressed here -- its magnitude |p| is
    # captured separately by the scalar `momentum` (log10) head -- so each
    # non-sentinel row is L2-normalized to unit length. Unlike the separate
    # px/py/pz scalars, this rides the geometric augmentations as a single vector
    # (a rotation mixes the components, so they must travel together); rotation
    # and flip preserve unit norm. The all-(-1) sentinel for undefined momentum
    # (e.g. LED) is left untouched so downstream sentinel masking still fires,
    # and is preserved by the transforms.
    momentum_vec = np.stack([data_px, data_py, data_pz], axis=1).astype(np.float32)
    valid = ~(momentum_vec == -1).all(axis=1)
    if valid.any():
        nrm = np.linalg.norm(momentum_vec[valid], axis=1, keepdims=True)
        momentum_vec[valid] = momentum_vec[valid] / np.clip(nrm, 1e-9, None)
    data_dict["momentum_vec"] = momentum_vec
    data_dict["true_energy"] = data_true_energy.astype(np.float32)[:, None]
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
