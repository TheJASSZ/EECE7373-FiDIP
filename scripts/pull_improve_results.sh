#!/usr/bin/env bash
# IMPROVE stream: pull ONLY the small per-run JSONs (run.json, metrics_*.json, val_curve.json,
# registry_row.csv) of R1xx runs into results/cluster_runs/ (Mac disk is nearly full; no preds/ckpts).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO/results/cluster_runs"; mkdir -p "$DEST"
ssh -o BatchMode=yes -o ConnectTimeout=15 explorer 'cd /scratch/purohitham.v/fidip/runs && \
  find R1* -maxdepth 1 -type f \( -name run.json -o -name "metrics_*.json" -o -name val_curve.json -o -name registry_row.csv -o -name cfg_opts.txt \) -print0 \
  | tar --null -czf - -T -' | tar -xzf - -C "$DEST"
echo "pulled improve run JSONs -> $DEST"
