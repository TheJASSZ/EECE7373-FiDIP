#!/bin/bash
#SBATCH --job-name=an_crosscheck
#SBATCH --partition=short
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/scratch/purohitham.v/fidip/slurm_logs/%x_%j.out
set -euo pipefail
R=/scratch/purohitham.v/fidip
module load miniconda3/25.9.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $R/envs/mmpose
cd $R/repo
mkdir -p $R/analysis/tables
python scripts/analysis/crosscheck_cocoeval_impls.py $R/data/splits/test100.json $R/analysis/tables/crosscheck_cocoeval_impls.json
