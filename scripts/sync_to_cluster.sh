#!/usr/bin/env bash
# Push the local repo (code, configs, docs) to the cluster copy at $R/repo.
# Uses tar-over-ssh because macOS ships openrsync (protocol 29), which fails against the
# cluster's rsync 3.2.5 ("unexpected end of file"). Never deletes remote files: cluster-side
# build products (e.g. compiled nms libs under third_party/.../lib) live only there.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR=/scratch/purohitham.v/fidip/repo
cd "$REPO"
COPYFILE_DISABLE=1 tar --no-xattrs -czf - \
  --exclude=./.git --exclude=./report --exclude=./results/figures --exclude=./results/cluster_runs \
  --exclude=.DS_Store --exclude=__pycache__ --exclude='*.pyc' --exclude=./.env . \
| ssh -o BatchMode=yes -o ConnectTimeout=15 explorer \
  "mkdir -p $REMOTE_DIR && tar -xzf - -C $REMOTE_DIR --warning=no-unknown-keyword --unlink-first"
# --unlink-first: each file is removed before it is re-created, so a job currently executing a script
# keeps reading the old inode instead of a file truncated/rewritten mid-run (D19).
echo "synced $REPO -> explorer:$REMOTE_DIR"
