#!/usr/bin/env bash
# Stage a PILArNet parquet root onto node-local NVMe (/lscratch) and print the
# env-var export the reader picks up. Compressed parquet read from local NVMe
# avoids the h5py random-read + HDF5 file-locking pain on Lustre entirely.
#
# Meant to run once per node before training. In a pimm launch config, drop the
# rsync into `setup:` and set the env var in `env:`, e.g.:
#
#   setup:
#     - bash scripts/pilarnet/stage_lscratch.sh /sdf/data/.../parquet v2
#   env:
#     PILARNET_PARQUET_ROOT_V2: /lscratch/${USER}/pilarnet/v2
#
# Usage: stage_lscratch.sh <src-parquet-root> [revision] [dest-root]
#   <src-parquet-root>  dir containing <revision>/ (the converter's --out-root)
#   [revision]          v1|v2|v3   (default: v2)
#   [dest-root]         default: /lscratch/$USER/pilarnet
set -eo pipefail

SRC_ROOT="${1:?usage: stage_lscratch.sh <src-parquet-root> [revision] [dest-root]}"
REV="${2:-v2}"
DEST_ROOT="${3:-/lscratch/${USER}/pilarnet}"

SRC="${SRC_ROOT%/}/${REV}"
DEST="${DEST_ROOT%/}/${REV}"

if [ ! -d "$SRC" ]; then
  echo "stage_lscratch: source not found: $SRC" >&2
  exit 1
fi

mkdir -p "$DEST"
echo "stage_lscratch: rsync $SRC/ -> $DEST/"
# -a preserve, --delete keep dest an exact mirror, whole-file (local NVMe, no
# need for the rsync delta algorithm).
rsync -a --delete --whole-file "$SRC/" "$DEST/"

echo "stage_lscratch: done. Set:"
echo "  export PILARNET_PARQUET_ROOT_${REV^^}=\"$DEST\""
