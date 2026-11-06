#!/bin/bash
# IMPROVE stream: derive the real200 / syn1000 train subsets from $R/data/splits/train_infant.json
# (by images[].is_synthetic) into $R/data/improve_splits, then check that every image of every split
# used by the configs exists. CPU only, seconds.
#SBATCH --job-name=prep_data_mmpose
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
python $R/repo/mmpose_exps/tools/make_train_subsets.py $R/data/splits/train_infant.json $R/data/improve_splits
sha256sum $R/data/splits/*.json $R/data/improve_splits/*.json
python $R/repo/mmpose_exps/tools/check_data.py $R/repo/mmpose_exps/configs/syrip_hrnet-w48_dark-384x288.py
