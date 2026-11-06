#!/usr/bin/env bash
# Pull SMALL run artifacts (run.json, metrics_*.json, preds_*.json, train logs, small PNGs) from
# $R/runs to results/cluster_runs/ locally. Never pulls checkpoints (Mac disk is nearly full).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO/results/cluster_runs"
mkdir -p "$DEST"
ssh -o BatchMode=yes -o ConnectTimeout=15 explorer 'cd /scratch/purohitham.v/fidip/runs && \
  find . -type f \( -name "run.json" -o -name "metrics_*.json" -o -name "preds_*.json" -o -name "*.log" -o -name "*.csv" -o -name "*.png" -o -name "*.yaml" -o -name "*.py" \) -size -20M -print0 \
  | tar --null -czf - -T -' | tar -xzf - -C "$DEST"
echo "pulled small run artifacts -> $DEST"
