#!/bin/bash
# Upload the PILArNet v3 (px/py/pz) HDF5 dataset from S3DF to the gcloud site's
# GCS bucket, so Cloud Batch jobs can read it via the gcsfuse mount.
#
# The gcloud site sets PILARNET_DATA_ROOT_V3=/mnt/disks/gcs/pilarnet (==
# gs://lartpc-artifacts/pilarnet), and the H5 reader globs <root>/*<split>/*.h5,
# so the data must land under pilarnet/{train,val,test}/.
#
# In the S3DF source, the per-split .h5 files are SYMLINKS back to the flat
# shards in the parent dir; the _points.npy index caches are real files. gsutil
# rsync follows file symlinks by default (uploads the pointed-to content), so we
# rsync each split dir individually -- NOT the parent, which would upload the
# flat shards once and then re-upload the same bytes through the split symlinks.
#
# Sizes (dereferenced): train ~141G, val ~8.4G, test ~8.0G (~157G total).
# Run under tmux/screen; requires an authenticated gsutil on the submit host.
#
# Pass -n for a dry run (prints what would transfer without uploading).
set -euo pipefail

SRC=/sdf/data/neutrino/gregork/larnet/h5/reprocessed_v2_pxpypz_0715
DST=gs://lartpc-artifacts/pilarnet

DRY=""
if [ "${1:-}" = "-n" ]; then
  DRY="-n"
  echo "# dry run: no data will be uploaded"
fi

for split in train val test; do
  echo "# rsync $SRC/$split -> $DST/$split"
  gsutil -m rsync -r $DRY "$SRC/$split" "$DST/$split"
done

echo "# done. Verify shard sizes (should be multi-GB, not tiny symlink files):"
echo "#   gsutil ls -l $DST/train/ | head"
