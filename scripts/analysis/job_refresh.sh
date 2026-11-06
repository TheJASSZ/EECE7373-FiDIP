#!/usr/bin/env bash
# Idempotent refresh of every analysis artefact (run from the private snapshot by analysis.sbatch):
#   sbatch scripts/slurm/analysis.sbatch script scripts/analysis/job_refresh.sh [--no-released]
set -uo pipefail
R=${R:-/scratch/purohitham.v/fidip}; A=$R/analysis
step() { echo "=== $*"; "$@" || echo "STEP FAILED ($?): $*"; }
step python scripts/analysis_pipeline.py eval-all
step python scripts/analysis/crosscheck_upstream.py
if [ "${1:-}" != "--no-released" ]; then
  LABEL="FiDIP HRNet-W48 (released)" SLOT=1 step bash scripts/analysis/run_released.sh R001_fidip-w48_released
  LABEL="FiDIP MobileNetV2 (released)" SLOT=8 step bash scripts/analysis/run_released.sh R002_fidip-mobilenet_released
fi
step python scripts/analysis/build_manifests.py
for s in val400 test100; do
  step python eval/analyze.py models --manifest $A/manifests/compare_$s.json --outdir $A/figures/compare_$s --tables $A/tables/compare_$s
done
# training curves: label@slot#seed=RUN_ID for every run dir that exists
mm=(); for spec in "HRNet-W48 control@3#s1=R101_mmp-hrnetw48_base_s1" "ViTPose-L 256@2#s1=R102_mmp-vitpose-l256_base_s1" \
    "ViTPose-H 384@7#s1=R104_mmp-vitpose-h384_base_s1" "ViTPose-H 256@6#s1=R105_mmp-vitpose-h256_base_s1"; do
  [ -d $R/runs/${spec##*=} ] && mm+=("$spec"); done
[ ${#mm[@]} -gt 0 ] && step bash scripts/analysis/curves.sh mmpose_backbones "${mm[@]}"
fd=(); for i in 0 1 2; do
  for spec in "FiDIP lambda=5e-4@4#s$((i+1))=R01${i}_fidip-w48_lambda5e-4_s$((i+1))" "Fine-tune lambda=0@5#s$((i+1))=R01$((i+3))_fidip-w48_lambda0_s$((i+1))"; do
    [ -f $R/runs/${spec##*=}/epochs.csv ] && fd+=("$spec"); done; done
[ ${#fd[@]} -gt 0 ] && step bash scripts/analysis/curves.sh fidip_retrain "${fd[@]}"
echo "refresh done"
