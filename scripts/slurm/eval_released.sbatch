#!/bin/bash
# R001/R002: evaluate the released FiDIP checkpoints with the upstream test script (GT boxes, flip test).
#   test100         = official validate100 GT + SyRIP.zip images          (our canonical Test100)
#   validate_infant = same GT + syrip_for_train.zip images, the layout the upstream README prescribes
#                     (its test11.jpg is a copy of test24.jpg; kept to show the as-shipped number)
#   val400          = Val400 (validate500 minus Test100 and its duplicates; area fixed, D8)
#   validate500     = all 500 (validate500's own GT boxes, area fixed, D8)  [R001 only, analysis request]
# Usage: sbatch scripts/slurm/eval_released.sbatch
#SBATCH --job-name=rp_eval_released
#SBATCH --partition=sharing
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/purohitham.v/fidip/slurm_logs/%x_%j.out
set -euo pipefail
echo "job $SLURM_JOB_ID on $(hostname) at $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
R=/scratch/purohitham.v/fidip
cd $R/repo
bash scripts/eval_upstream.sh R001_fidip-w48_released experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml \
  $R/models/hrnet_fidip.pth '' test100 validate_infant val400 validate500
bash scripts/eval_upstream.sh R002_fidip-mobilenet_released experiments/coco/mobilenet/mobile_224x224_adam_lr1e-3_infant.yaml \
  $R/models/mobile_fidip.pth '' test100 validate_infant val400
echo "done at $(date)"
