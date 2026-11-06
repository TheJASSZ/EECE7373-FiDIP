#!/bin/bash
# IMPROVE stream: zero-shot (no fine-tuning) Val400 evaluation of the COCO checkpoints, i.e. each
# config's own `load_from`, with that config's test pipeline (GT boxes, flip test). This checks the
# eval path end to end and gives the "COCO model without fine-tune" reference on Val400.
# Required env: PAIRS="RUN_ID:repo/relative/config.py RUN_ID:config.py ..."
#SBATCH --job-name=im_zeroshot_mmpose
#SBATCH --partition=sharing
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/scratch/purohitham.v/fidip/slurm_logs/%x_%j.out
set -euo pipefail
: "${PAIRS:?set PAIRS}"
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
cd $R/mmpose
for pair in $PAIRS; do
  RUN_ID=${pair%%:*}; CONFIG=${pair#*:}; RUN_DIR=$R/runs/$RUN_ID
  echo "=== $RUN_ID  $CONFIG  $(date -Is)"
  test -f $RUN_DIR/run.json && { echo "skip: finalized"; continue; }
  mkdir -p $RUN_DIR
  cp $REPO/$CONFIG $RUN_DIR/config_src.py
  cp $REPO/GIT_INFO_mmpose.json $RUN_DIR/git_info.json
  python -c "from mmengine.config import Config; Config.fromfile('$REPO/$CONFIG').dump('$RUN_DIR/config.py')"
  CKPT=$(python -c "from mmengine.config import Config; print(Config.fromfile('$RUN_DIR/config.py').load_from)")
  python $REPO/mmpose_exps/tools/eval_ckpt.py $RUN_DIR/config.py $CKPT --split val400 --out-dir $RUN_DIR --run-id $RUN_ID
  python $REPO/eval/eval_coco.py --gt $R/data/splits/val400.json --pred $RUN_DIR/preds_val400.json \
    --out $RUN_DIR/metrics_val400.json --per-image $RUN_DIR/per_image_val400.json --run-id $RUN_ID
  python $REPO/mmpose_exps/tools/finalize_run.py $RUN_DIR --run-id $RUN_ID --config $RUN_DIR/config_src.py \
    --config-name $(basename $CONFIG) --seed 0 --eval-only-ckpt $CKPT --notes "zero-shot COCO checkpoint"
done
echo "done $(date -Is)"
