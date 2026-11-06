#!/bin/bash
# GPU sanity check of $R/envs/mmpose: versions, arch list, mmcv CUDA ops, checkpoint key match
# and a timed train step per config. Submit with e.g.
#   sbatch -p gpu --gres=gpu:h200:1 verify_env_mmpose.sbatch      (sm_90)
#   sbatch -p sharing --gres=gpu:a100:1 verify_env_mmpose.sbatch  (sm_80)
#SBATCH --job-name=im_verify_env_mmpose
#SBATCH --partition=sharing
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/scratch/purohitham.v/fidip/slurm_logs/%x_%j.out
set -euo pipefail
R=/scratch/purohitham.v/fidip
# D19: run from a per-job code snapshot (a concurrent sync must never rewrite scripts under a running job)
SNAP=$R/snapshots/${SLURM_JOB_ID:-manual_$$}
mkdir -p $SNAP && rsync -a --exclude=.git $R/repo/ $SNAP/ || { echo "D19 snapshot failed"; exit 1; }
export REPO=$SNAP
module load miniconda3/25.9.1 cuda/12.1.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $R/envs/mmpose
export PYTHONNOUSERSITE=1   # never let ~/.local site-packages shadow the env (orchestrator note)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
C=$REPO/mmpose_exps/configs
mkdir -p $R/logs/verify
cd $R/mmpose
python $REPO/mmpose_exps/tools/verify_env.py --out $R/logs/verify/verify_${SLURM_JOB_ID}.json \
  $C/syrip_hrnet-w48_dark-384x288.py:32 \
  $C/syrip_vitpose-l_udp-256x192.py:64 \
  $C/syrip_vitpose-l_udp-384x288.py:64 \
  $C/syrip_vitpose-h_udp-256x192.py:64 \
  $C/syrip_vitpose-h_udp-384x288.py:64 \
  $C/syrip_vitpose-h_udp-384x288.py:32
