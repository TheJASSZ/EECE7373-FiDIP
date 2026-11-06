#!/bin/bash
# Download released FiDIP checkpoints + COCO DarkPose HRNet-W48 into $R/models (CPU job, needs internet).
# Usage: sbatch scripts/slurm/get_models.sbatch [--list-only]
#SBATCH --job-name=rp_get_models
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/purohitham.v/fidip/slurm_logs/%x_%j.out
set -euo pipefail
echo "job $SLURM_JOB_ID on $(hostname) at $(date)"
export PYTHONNOUSERSITE=1
R=/scratch/purohitham.v/fidip
cd $R/repo
$R/envs/fidip/bin/python -I scripts/get_models.py --out $R/models "$@"
ls -la $R/models $R/models/coco 2>/dev/null || true
