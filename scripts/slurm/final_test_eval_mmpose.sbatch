#!/bin/bash
# FINAL Test100 evaluation of ONE pre-registered MMPose run (GT boxes, flip test as configured).
#   * refuses unless RUN_ID has a row in results/PREREGISTRATION.md (checked here and again in
#     mmpose_exps/tools/eval_ckpt.py), per CLAUDE.md;
#   * evaluates exactly the checkpoint recorded in run.json (the Val400-selected epoch) and
#     verifies its SHA256 against run.json before running;
#   * writes preds_test100.json and metrics_test100.json (eval/eval_coco.py, paper + fixed).
# Required env: RUN_ID.   sbatch --export=ALL,RUN_ID=R1xx_... scripts/slurm/final_test_eval_mmpose.sbatch
#SBATCH --job-name=im_final_test_mmpose
#SBATCH --partition=sharing
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/purohitham.v/fidip/slurm_logs/%x_%j.out
set -euo pipefail
: "${RUN_ID:?set RUN_ID}"
R=/scratch/purohitham.v/fidip
REPO=$R/repo
RUN_DIR=$R/runs/$RUN_ID
PREREG=$REPO/results/PREREGISTRATION.md
grep -E "^\|.*${RUN_ID}" $PREREG >/dev/null || { echo "REFUSED: $RUN_ID not in $PREREG"; exit 2; }
test ! -e $RUN_DIR/preds_test100.json || { echo "REFUSED: $RUN_DIR/preds_test100.json exists (one shot only)"; exit 3; }
module load miniconda3/25.9.1 cuda/12.1.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $R/envs/mmpose
# TF32 tensor-core matmuls for every IMPROVE run and eval (cuDNN convs already default to TF32 in
# PyTorch 2.1, so this puts the ViT matmuls on the same footing as HRNet's convs). In run.json.
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-1}
export PYTHONPATH=$REPO/mmpose_exps:${PYTHONPATH:-}   # custom modules (mmpose_exps/syrip_da)
read CKPT SHA < <(python -c "import json;d=json.load(open('$RUN_DIR/run.json'));print(d['checkpoint'],d['checkpoint_sha256'])")
echo "run=$RUN_ID ckpt=$CKPT job=${SLURM_JOB_ID:-} node=$(hostname) start=$(date -Is)"
echo "$SHA  $CKPT" | sha256sum -c -
cd $R/mmpose
python $REPO/mmpose_exps/tools/eval_ckpt.py $RUN_DIR/config.py $CKPT --split test100 \
  --out-dir $RUN_DIR --run-id $RUN_ID
python $REPO/eval/eval_coco.py --gt $R/data/splits/test100.json --pred $RUN_DIR/preds_test100.json \
  --out $RUN_DIR/metrics_test100.json --per-image $RUN_DIR/per_image_test100.json --run-id $RUN_ID
echo "done $(date -Is)"
