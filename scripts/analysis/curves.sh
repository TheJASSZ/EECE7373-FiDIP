#!/usr/bin/env bash
# Training curves for a set of runs. Usage (inside sbatch analysis.sbatch script):
#   scripts/analysis/curves.sh NAME "label#s1=RUN_ID" "label#s2=RUN_ID" ...
set -euo pipefail
R=${R:-/scratch/purohitham.v/fidip}
NAME=$1; shift
specs=()
for s in "$@"; do lab=${s%=*}; rid=${s##*=}; specs+=("$lab=$R/runs/$rid"); done
mkdir -p $R/analysis/figures/curves $R/analysis/tables
python eval/analyze.py curves --runs "${specs[@]}" --name "$NAME" --outdir $R/analysis/figures/curves --tables $R/analysis/tables
