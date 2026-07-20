#!/usr/bin/env python
"""Precompute per-event point-count sidecars for PILArNet HDF5 shards.

``PILArNetH5Dataset._build_index`` (pimm/datasets/pilarnet/h5.py) filters events
by point count at startup. For each shard ``foo.h5`` it looks for a sibling
``foo_points.npy``; if that is missing it falls back to **reading every event's
point array on the fly** to count points -- tens of seconds per multi-GB shard,
paid on *every* run.

This script builds those ``*_points.npy`` sidecars once (in parallel across
shards), so subsequent runs load a tiny array instead of rescanning the data.
The point count matches the loader exactly: ``f['point'][i].size // 8`` (the
flat point record is 8 values wide).

Examples
--------
Build for the split subdirs the loader reads (recurses into train/val/test)::

    python scripts/pilarnet/build_points_index.py \\
        /sdf/data/neutrino/gregork/larnet/h5/reprocessed_v2_pxpypz_0715

Build for a single flat directory (non-recursive)::

    python scripts/pilarnet/build_points_index.py SRC --no-recurse

Sidecars already present are skipped unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - h5py is a hard runtime dep of pimm
    sys.exit("h5py is required; run inside the pimm environment (e.g. `uv run`).")


def sidecar_path(h5_file: str) -> str:
    # Match the loader's convention exactly (h5.py: h5_file.replace(...)).
    return h5_file.replace(".h5", "_points.npy")


def build_one(h5_file: str, force: bool) -> tuple[str, int, str]:
    """Write ``<shard>_points.npy`` for one shard. Returns (path, n_events, note)."""
    out = sidecar_path(h5_file)
    if os.path.exists(out) and not force:
        return (out, -1, "skip (exists)")
    with h5py.File(h5_file, "r", libver="latest", swmr=True) as f:
        if "point" not in f:
            return (out, 0, "ERROR: no 'point' dataset")
        n = f["point"].shape[0]
        npoints = np.empty(n, dtype=np.int64)
        pts = f["point"]
        for i in range(n):
            # Same count the loader uses: flat record is 8 wide (x,y,z,e,...).
            npoints[i] = pts[i].size // 8
    # Write atomically (tmp + rename) so a killed run never leaves a half file.
    # np.save appends .npy unless the name already ends in it, so give tmp a
    # .npy suffix and rename that exact path into place.
    tmp = out + ".tmp.npy"
    np.save(tmp, npoints)
    os.replace(tmp, out)
    return (out, int(n), "built")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Precompute *_points.npy sidecars for PILArNet shards.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("root", help="Directory containing .h5 shards (searched recursively by default).")
    p.add_argument("--no-recurse", action="store_true", help="Only the top-level dir, no subdirs.")
    p.add_argument("--force", action="store_true", help="Rebuild sidecars that already exist.")
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel worker processes. 0 = min(#shards, cpu_count). One shard per worker.",
    )
    p.add_argument("--dry-run", action="store_true", help="List shards that would be built and exit.")
    args = p.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        sys.exit(f"root is not a directory: {root}")

    pattern = os.path.join(root, "*.h5") if args.no_recurse else os.path.join(root, "**", "*.h5")
    files = sorted(glob.glob(pattern, recursive=not args.no_recurse))
    # Never treat a sidecar-less run's own output as input.
    files = [f for f in files if not f.endswith("_points.npy")]
    if not files:
        sys.exit(f"No .h5 shards found under {root}")

    todo = [f for f in files if args.force or not os.path.exists(sidecar_path(f))]
    print(f"{len(files)} shards found; {len(todo)} need sidecars"
          + (" (--force)" if args.force else ""))
    if args.dry_run:
        for f in todo:
            print(f"  would build {sidecar_path(f)}")
        return
    if not todo:
        print("Nothing to do.")
        return

    workers = args.workers or min(len(todo), os.cpu_count() or 1)
    print(f"Building with {workers} workers ...")
    errors = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_one, f, args.force): f for f in todo}
        for fut in as_completed(futs):
            out, n, note = fut.result()
            print(f"  [{note}] {out}" + (f"  ({n} events)" if n >= 0 else ""))
            if note.startswith("ERROR"):
                errors += 1

    print("Done." + (f" {errors} shard(s) errored." if errors else ""))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
