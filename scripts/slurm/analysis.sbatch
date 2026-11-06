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
# Run from a private snapshot: other streams re-extract $R/repo while jobs run, and bash reads scripts
# incrementally, so an in-place rewrite of a running script silently ends it (seen 2026-11-06, jobs
# 11664900/11664980: exit 0 mid-script). The snapshot also pins one code version per job.
SNAP=$R/analysis/snapshots/${SLURM_JOB_ID:-manual_$$}
mkdir -p $SNAP/results
cp -r $R/repo/eval $R/repo/scripts $SNAP/
cp $R/repo/results/*.md $R/repo/results/*.csv $R/repo/results/*.json $SNAP/results/ 2>/dev/null || true
cp $R/repo/.git_tree_analysis $SNAP/ 2>/dev/null || true
cd $SNAP
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -Is) snapshot=$SNAP tree=$(cat .git_tree_analysis 2>/dev/null || echo n/a)"
mode=${1:-eval-all}; shift || true
case "$mode" in
  eval-all) python scripts/analysis_pipeline.py eval-all "$@" ;;
  figures)  M=$1; OUT=$R/analysis/figures/$(basename "$M" .json); TAB=$R/analysis/tables
            python eval/analyze.py models --manifest "$M" --outdir "$OUT" --tables "$TAB" ;;
  script)   cp "$1" $SNAP/job_script.sh; bash $SNAP/job_script.sh "${@:2}" ;;
  *) echo "unknown mode $mode"; exit 2 ;;
esac
echo "end=$(date -Is)"
