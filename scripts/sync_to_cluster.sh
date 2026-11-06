#!/usr/bin/env bash
# Push the local repo (code, configs, docs) to the cluster copy at $R/repo.
#
# Why this shape:
# - Tar-over-ssh, because macOS ships openrsync (protocol 29), which fails against the cluster's
#   rsync 3.2.5 with "unexpected end of file".
# - Never deletes remote files. Cluster-side build products (e.g. compiled nms libs) live only there.
# - Running jobs must never see a script change underneath them (D19). bash reads scripts
#   incrementally, so overwriting a file in place corrupts running jobs. We therefore unpack into a
#   private staging dir and let the cluster's own rsync copy it over. rsync writes each file to a
#   temp name and renames it, so a running job keeps reading the old inode.
#   (`tar --unlink-first` was tried first; it also tries to unlink non-empty directories and fails.)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
R=/scratch/purohitham.v/fidip
REMOTE_DIR=$R/repo
cd "$REPO"
COPYFILE_DISABLE=1 tar --no-xattrs -czf - \
  --exclude=./.git --exclude=./report --exclude=./results/figures --exclude=./results/cluster_runs \
  --exclude=.DS_Store --exclude=__pycache__ --exclude='*.pyc' --exclude=./.env . \
| ssh -o BatchMode=yes -o ConnectTimeout=15 explorer \
  "set -e; mkdir -p $REMOTE_DIR $R/.sync_staging; S=\$(mktemp -d $R/.sync_staging/XXXXXX); \
   tar -xzf - -C \"\$S\" --warning=no-unknown-keyword; rsync -a \"\$S\"/ $REMOTE_DIR/; rm -r -- \"\$S\""
echo "synced $REPO -> explorer:$REMOTE_DIR (staged copy + atomic per-file rename)"
