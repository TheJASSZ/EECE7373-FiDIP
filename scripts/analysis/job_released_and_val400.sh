#!/usr/bin/env bash
# One CPU job: evaluate everything new, cross-check upstream, released-model figure sets, Val400 comparison.
#   sbatch scripts/slurm/analysis.sbatch script scripts/analysis/job_released_and_val400.sh
set -uo pipefail
R=${R:-/scratch/purohitham.v/fidip}; A=$R/analysis
step() { echo "=== $*"; "$@" || echo "STEP FAILED ($?): $*"; }
step python scripts/analysis_pipeline.py eval-all
step python scripts/analysis/crosscheck_upstream.py
LABEL="FiDIP HRNet-W48 (released)" SLOT=1 step bash scripts/analysis/run_released.sh R001_fidip-w48_released
LABEL="FiDIP MobileNetV2 (released)" SLOT=8 step bash scripts/analysis/run_released.sh R002_fidip-mobilenet_released
step python scripts/analysis/build_manifests.py
for s in val400 test100; do
  step python eval/analyze.py models --manifest $A/manifests/compare_$s.json --outdir $A/figures/compare_$s \
       --tables $A/tables/compare_$s
done
step bash scripts/analysis/curves.sh mmpose_screen "HRNet-W48 control@3#s1=R101_mmp-hrnetw48_base_s1" \
     "ViTPose-L 256@2#s1=R102_mmp-vitpose-l256_base_s1" "HRNet-W48 LR 2e-4@6#s1=R106_mmp-hrnetw48_lr2e-4_s1"
echo "job done"
