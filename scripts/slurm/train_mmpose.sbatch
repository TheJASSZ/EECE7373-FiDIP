#!/bin/bash
# IMPROVE stream: ONE fine-tuning run (body: scripts/run_one_mmpose.sh). Never touches Test100.
# Required env: RUN_ID, CONFIG (repo-relative). Optional: SEED (default 1), CFG_OPTS.
# Submit via scripts/launch_mmpose.sh (job name im_<RUN_ID>). D14 budget: target A100/L40S/A6000,
# not H100/H200 (mmcv 2.1.0 has no sm_90 kernels). For several runs on one GPU use
# train_pack_mmpose.sbatch. Resubmitting the same RUN_ID resumes from $RUN_DIR/last_checkpoint.
#SBATCH --job-name=im_train_mmpose
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/purohitham.v/fidip/slurm_logs/%x_%j.out
set -euo pipefail
: "${RUN_ID:?set RUN_ID}" "${CONFIG:?set CONFIG}"
R=/scratch/purohitham.v/fidip
REPO=$R/repo
module load miniconda3/25.9.1 cuda/12.1.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $R/envs/mmpose
export PYTHONNOUSERSITE=1   # never let ~/.local site-packages shadow the env (orchestrator note)
# TF32 tensor-core matmuls for every IMPROVE run and eval (cuDNN convs already default to TF32 in
# PyTorch 2.1, so this puts the ViT matmuls on the same footing as HRNet's convs). In run.json.
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-1}
export PYTHONPATH=$REPO/mmpose_exps:${PYTHONPATH:-}   # custom modules (mmpose_exps/syrip_da)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
bash $REPO/scripts/run_one_mmpose.sh "$RUN_ID" "$CONFIG" "${SEED:-1}" "${CFG_OPTS:-}"
