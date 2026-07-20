#!/usr/bin/env python
"""Split a flat directory of PILArNet HDF5 shards into train/val/test.

``PILArNetH5Dataset`` discovers shards by globbing ``{data_root}/*{split}/*.h5``
(see ``pimm/datasets/pilarnet/h5.py``), so the shards must live in subdirectories
whose names end in the split word (``train`` / ``val`` / ``test``). Reprocessed
datasets (e.g. ``reprocessed_v2_pxpypz_0715/``) instead ship the ``.h5`` files
flat in one directory, which matches *zero* files for any split.

This script builds the expected ``train/``, ``val/``, ``test/`` subdirectories
next to (or inside) the source directory and populates them -- by default with
**symlinks**, so no bytes are copied (these shards are tens of GB each).

Splitting is done at the *file* level: whole shards are assigned to a split.
Files are shuffled deterministically (``--seed``) and assigned greedily so the
per-split share of *events* (not just file count) tracks the requested ratios.
This keeps events from a single shard from straddling splits, which matters if a
shard is internally ordered.

Examples
--------
Default 90/5/5 split, symlinks, subdirs created inside the source dir::

    python scripts/pilarnet/split_dataset.py \\
        /sdf/data/neutrino/gregork/larnet/h5/reprocessed_v2_pxpypz_0715

Then train with ``data.*.data_root`` pointing at that same directory::

    pimm submit --site s3df \\
        --train.config panda/panseg/detector-v5-pt-v3m2-ft-joint-pxpypz-fft \\
        -- data.train.data_root=/sdf/data/neutrino/gregork/larnet/h5/reprocessed_v2_pxpypz_0715 \\
           data.val.data_root=/sdf/data/neutrino/gregork/larnet/h5/reprocessed_v2_pxpypz_0715

Custom ratios / separate output dir / hard copies::

    python scripts/pilarnet/split_dataset.py SRC \\
        --out /sdf/data/neutrino/gregork/pimm_data/pxpypz_split \\
        --ratios 0.8 0.1 0.1 --copy
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import shutil
import sys

try:
    import h5py
except ImportError:  # pragma: no cover - h5py is a hard runtime dep of pimm
    sys.exit("h5py is required; run inside the pimm environment (e.g. `uv run`).")

SPLITS = ("train", "val", "test")


def count_events(path: str) -> int:
    """Number of events in a shard (the length of the ``point`` dataset)."""
    with h5py.File(path, "r") as f:
        if "point" not in f:
            raise KeyError(f"{path}: no 'point' dataset -- not a PILArNet shard?")
        return int(f["point"].shape[0])


def assign_files(files_with_counts, ratios, seed):
    """Greedily assign whole shards to splits to track target event ratios.

    Returns a dict ``{split: [paths]}``. Shards are visited largest-first after a
    seeded shuffle tiebreak, and each is placed in the split currently furthest
    *below* its event target -- a simple longest-processing-time style balance
    that avoids one split hogging every large shard.
    """
    total_events = sum(n for _, n in files_with_counts)
    targets = {s: r * total_events for s, r in zip(SPLITS, ratios)}
    assigned = {s: [] for s in SPLITS}
    got = {s: 0 for s in SPLITS}

    rng = random.Random(seed)
    order = list(files_with_counts)
    rng.shuffle(order)
    # Largest shards first so big lumps are placed while balancing headroom.
    order.sort(key=lambda fc: fc[1], reverse=True)

    for path, n in order:
        # Pick the split with the most remaining room (target - got), but never
        # a split whose target is 0 (ratio 0 -> stays empty).
        candidates = [s for s in SPLITS if targets[s] > 0]
        best = max(candidates, key=lambda s: targets[s] - got[s])
        assigned[best].append(path)
        got[best] += n

    return assigned, got, total_events


def main() -> None:
    p = argparse.ArgumentParser(
        description="Split flat PILArNet .h5 shards into train/val/test subdirs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("src", help="Directory containing the flat *.h5 shards.")
    p.add_argument(
        "--out",
        default=None,
        help="Where to create train/val/test subdirs. Defaults to SRC itself "
        "(subdirs created in place; the flat shards stay put and are ignored by "
        "the split-aware glob).",
    )
    p.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=(0.9, 0.05, 0.05),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Event-fraction targets for train/val/test. Default: 0.9 0.05 0.05.",
    )
    p.add_argument(
        "--glob",
        default="*.h5",
        help="Shard glob within SRC. Default: '*.h5'.",
    )
    p.add_argument("--seed", type=int, default=0, help="Shuffle seed. Default: 0.")
    p.add_argument(
        "--copy",
        action="store_true",
        help="Copy shards instead of symlinking (slow; ~178 GB for the full set).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned assignment without creating anything.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing links/files in the split subdirs.",
    )
    args = p.parse_args()

    src = os.path.abspath(args.src)
    if not os.path.isdir(src):
        sys.exit(f"src is not a directory: {src}")
    if abs(sum(args.ratios) - 1.0) > 1e-6:
        sys.exit(f"ratios must sum to 1.0, got {args.ratios} (sum {sum(args.ratios)})")

    out = os.path.abspath(args.out) if args.out else src

    files = sorted(glob.glob(os.path.join(src, args.glob)))
    # Avoid re-splitting files that already live under a split subdir.
    files = [f for f in files if os.path.basename(os.path.dirname(f)) not in SPLITS]
    if not files:
        sys.exit(f"No shards match {args.glob!r} in {src}")

    print(f"Counting events in {len(files)} shards ...")
    files_with_counts = [(f, count_events(f)) for f in files]

    assigned, got, total = assign_files(files_with_counts, args.ratios, args.seed)

    print(f"\nTotal: {total} events across {len(files)} shards")
    print(f"{'split':>6} {'files':>6} {'events':>12} {'actual':>8} {'target':>8}")
    for s, r in zip(SPLITS, args.ratios):
        frac = got[s] / total if total else 0.0
        print(f"{s:>6} {len(assigned[s]):>6} {got[s]:>12} {frac:>8.3f} {r:>8.3f}")

    if args.dry_run:
        print("\n[dry-run] planned assignment:")
        for s in SPLITS:
            for f in assigned[s]:
                print(f"  {s}/  <- {os.path.basename(f)}")
        return

    link_or_copy = shutil.copy2 if args.copy else os.symlink
    verb = "Copying" if args.copy else "Linking"
    for s in SPLITS:
        if not assigned[s]:
            continue
        dst_dir = os.path.join(out, s)
        os.makedirs(dst_dir, exist_ok=True)
        for f in assigned[s]:
            dst = os.path.join(dst_dir, os.path.basename(f))
            if os.path.lexists(dst):
                if not args.force:
                    sys.exit(f"exists (use --force): {dst}")
                os.remove(dst)
            link_or_copy(f, dst)
            # Carry over the fast-index sidecar if present, so the loader skips
            # the on-the-fly point count.
            sidecar = f.replace(".h5", "_points.npy")
            if os.path.exists(sidecar):
                dst_sc = os.path.join(dst_dir, os.path.basename(sidecar))
                if os.path.lexists(dst_sc):
                    os.remove(dst_sc)
                link_or_copy(sidecar, dst_sc)
        print(f"{verb} {len(assigned[s])} shards -> {dst_dir}")

    print(f"\nDone. Set data.*.data_root={out}")


if __name__ == "__main__":
    main()
