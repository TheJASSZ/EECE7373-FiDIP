#!/bin/bash
# IMPROVE stream: eval-only Val400 re-evaluation of a finished run's selected checkpoint with a
# changed test-time setting (e.g. flip-TTA off). Writes preds_val400<TAG>.json and
# metrics_val400<TAG>.json (eval/eval_coco.py) next to the run's other files.
# Required env: RUN_ID, TAG (e.g. _noflip). Optional: EVAL_ARGS (e.g. "--no-flip").
#SBATCH --job-name=im_eval_val400_mmpose
#SBATCH --partition=sharing
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/purohitham.v/fidip/slurm_logs/%x_%j.out
set -euo pipefail
: "${RUN_ID:?set RUN_ID}" "${TAG:?set TAG}"
R=/scratch/purohitham.v/fidip
REPO=$R/repo
RUN_DIR=$R/runs/$RUN_ID
module load miniconda3/25.9.1 cuda/12.1.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $R/envs/mmpose
export PYTHONNOUSERSITE=1   # never let ~/.local site-packages shadow the env (orchestrator note)
# TF32 tensor-core matmuls for every IMPROVE run and eval (cuDNN convs already default to TF32 in
# PyTorch 2.1, so this puts the ViT matmuls on the same footing as HRNet's convs). In run.json.
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-1}
export PYTHONPATH=$REPO/mmpose_exps:${PYTHONPATH:-}   # custom modules (mmpose_exps/syrip_da)
CKPT=$(python -c "import json;print(json.load(open('$RUN_DIR/run.json'))['checkpoint'])")
echo "run=$RUN_ID tag=$TAG args='${EVAL_ARGS:-}' ckpt=$CKPT job=${SLURM_JOB_ID:-} node=$(hostname)"
cd $R/mmpose
python $REPO/mmpose_exps/tools/eval_ckpt.py $RUN_DIR/config.py $CKPT --split val400 \
  --out-dir $RUN_DIR --tag $TAG --run-id $RUN_ID ${EVAL_ARGS:-}
python $REPO/eval/eval_coco.py --gt $R/data/splits/val400.json --pred $RUN_DIR/preds_val400$TAG.json \
  --out $RUN_DIR/metrics_val400$TAG.json --per-image $RUN_DIR/per_image_val400$TAG.json --run-id $RUN_ID
