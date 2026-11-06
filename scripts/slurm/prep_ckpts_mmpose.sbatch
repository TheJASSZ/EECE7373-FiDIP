#!/bin/bash
# Derive the 384x288 ViTPose checkpoints: resize backbone.pos_embed 16x12 -> 24x18 (bicubic) for
# ViTPose-L and ViTPose-H (see mmpose_exps/tools/resize_vit_pos_embed.py for why). CPU only.
#SBATCH --job-name=im_prep_ckpts_mmpose
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=/scratch/purohitham.v/fidip/slurm_logs/%x_%j.out
set -euo pipefail
R=/scratch/purohitham.v/fidip
module load miniconda3/25.9.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $R/envs/mmpose
M=$R/models/mmpose
for n in td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314 \
         td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314; do
  python $R/repo/mmpose_exps/tools/resize_vit_pos_embed.py $M/$n.pth $M/derived/${n}_posembed24x18.pth \
    --src-grid 16 12 --dst-grid 24 18
done
(cd $M/derived && sha256sum *.pth | tee SHA256SUMS)
