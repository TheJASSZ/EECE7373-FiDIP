#!/bin/bash
# CPU job: verify + unzip SyRIP, build the upstream layout and the splits.
# Usage: sbatch scripts/slurm/get_data.sbatch [stage ...]   (default stages: verify unzip layout splits)
#SBATCH --job-name=fidip_get_data
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/purohitham.v/fidip/slurm_logs/%x_%j.out
set -euo pipefail
echo "job $SLURM_JOB_ID on $(hostname) at $(date)"
cd /scratch/purohitham.v/fidip/repo
bash scripts/get_data.sh "$@"
