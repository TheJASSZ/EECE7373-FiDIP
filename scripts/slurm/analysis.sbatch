#!/bin/bash
# Analysis stream CPU job. Usage:
#   sbatch scripts/slurm/analysis.sbatch eval-all                 # evaluate every new/changed preds_*.json
#   sbatch scripts/slurm/analysis.sbatch figures <manifest.json>   # analyze.py models + overlays for a manifest
#   sbatch scripts/slurm/analysis.sbatch script <file.sh>          # run an analysis shell script (logged)
#SBATCH --job-name=an_analysis
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/purohitham.v/fidip/slurm_logs/%x_%j.out
set -euo pipefail
export PYTHONNOUSERSITE=1   # D16: keep ~/.local site-packages out of the conda env
export R=/scratch/purohitham.v/fidip
module load miniconda3/25.9.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $R/envs/analysis
cd $R/repo
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is) tree=$(cat .git_tree_analysis 2>/dev/null || echo n/a)"
mode=${1:-eval-all}; shift || true
case "$mode" in
  eval-all) python scripts/analysis_pipeline.py eval-all "$@" ;;
  figures)  M=$1; OUT=$R/analysis/figures/$(basename "$M" .json); TAB=$R/analysis/tables
            python eval/analyze.py models --manifest "$M" --outdir "$OUT" --tables "$TAB" ;;
  script)   bash "$1" ;;
  *) echo "unknown mode $mode"; exit 2 ;;
esac
echo "end=$(date -Is)"
