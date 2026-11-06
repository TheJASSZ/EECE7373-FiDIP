#!/bin/bash
# Analysis helper that needs the mmpose env (xtcocotools): run a python script from a private snapshot.
#   sbatch scripts/slurm/an_mmpose_py.sbatch scripts/analysis/sensitivity_extra.py R001_fidip-w48_released
#SBATCH --job-name=an_mmpose_py
#SBATCH --partition=short
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/scratch/purohitham.v/fidip/slurm_logs/%x_%j.out
set -euo pipefail
export PYTHONNOUSERSITE=1
export R=/scratch/purohitham.v/fidip
module load miniconda3/25.9.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $R/envs/mmpose
SNAP=$R/analysis/snapshots/${SLURM_JOB_ID:-manual_$$}
mkdir -p $SNAP/results && cp -r $R/repo/eval $R/repo/scripts $SNAP/ && cp $R/repo/results/*.json $R/repo/results/*.md $R/repo/results/*.csv $SNAP/results/ 2>/dev/null || true
cd $SNAP
python "$@"
