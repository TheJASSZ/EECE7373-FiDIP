#!/usr/bin/env bash
# Pull the analysis stream's SMALL outputs from $R/analysis to the local repo:
#   eval metrics -> results/analysis_eval/<RUN_ID>/metrics_<split>.json (no per-image files)
#   figures      -> results/figures/  (png/pdf only)
#   tables       -> results/tables/   (csv/json)
#   bootstrap    -> results/tables/bootstrap_*.json
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO/results/analysis_eval" "$REPO/results/figures" "$REPO/results/tables"
ssh -o BatchMode=yes -o ConnectTimeout=15 explorer 'cd /scratch/purohitham.v/fidip/analysis 2>/dev/null && \
  find eval -name "metrics_*.json" -size -5M -print0 | tar --null -czf - -T -' | tar -xzf - -C "$REPO/results/analysis_eval" --strip-components=1
ssh -o BatchMode=yes -o ConnectTimeout=15 explorer 'cd /scratch/purohitham.v/fidip/analysis/figures 2>/dev/null && \
  find . \( -name "*.png" -o -name "*.pdf" -o -name "*.json" \) -size -20M -print0 | tar --null -czf - -T -' | tar -xzf - -C "$REPO/results/figures"
ssh -o BatchMode=yes -o ConnectTimeout=15 explorer 'cd /scratch/purohitham.v/fidip/analysis && \
  find tables bootstrap \( -name "*.csv" -o -name "*.json" \) -size -5M -print0 2>/dev/null | tar --null -czf - -T -' \
  | tar -xzf - -C "$REPO/results/tables" --strip-components=1
ssh -o BatchMode=yes -o ConnectTimeout=15 explorer 'cat /scratch/purohitham.v/fidip/analysis/eval/index.csv 2>/dev/null' \
  > "$REPO/results/analysis_eval/index.csv" || true
du -sh "$REPO/results/figures" "$REPO/results/tables" "$REPO/results/analysis_eval" | cat
