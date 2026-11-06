#!/bin/bash
# IMPROVE stream: fine-tune one MMPose config on SyRIP train (Val400 selection every epoch), then
#   1) re-run the selected checkpoint (best Val400 coco/AP epoch) on Val400 -> preds_val400.json
#   2) score it with the project evaluator eval/eval_coco.py        -> metrics_val400.json
#   3) write run.json / val_curve.json / registry_row.csv            (finalize_run.py)
# Never touches Test100 (see final_test_eval_mmpose.sbatch).
# Required env: RUN_ID, CONFIG (repo-relative). Optional: SEED (default 1), CFG_OPTS (extra
# --cfg-options for tools/train.py, e.g. "train_cfg.max_epochs=2" for a smoke test).
#   sbatch -J R102 -p gpu --gres=gpu:h200:1 -t 02:00:00 \
#     --export=ALL,RUN_ID=R102_mmp-vitpose-l_base_s1,CONFIG=mmpose_exps/configs/x.py,SEED=1 \
#     scripts/slurm/train_mmpose.sbatch
# Resubmitting the same RUN_ID resumes from $RUN_DIR/last_checkpoint (8 h partition limit).
#SBATCH --job-name=train_mmpose
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/purohitham.v/fidip/slurm_logs/%x_%j.out
set -euo pipefail
: "${RUN_ID:?set RUN_ID}" "${CONFIG:?set CONFIG}"
SEED=${SEED:-1}
R=/scratch/purohitham.v/fidip
REPO=$R/repo
RUN_DIR=$R/runs/$RUN_ID
module load miniconda3/25.9.1 cuda/12.1.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $R/envs/mmpose
# TF32 tensor-core matmuls for every IMPROVE run and eval (cuDNN convs already default to TF32 in
# PyTorch 2.1, so this puts the ViT matmuls on the same footing as HRNet's convs). In run.json.
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-1}
export PYTHONHASHSEED=$SEED
STARTED=$(date -Is)
echo "run=$RUN_ID config=$CONFIG seed=$SEED cfg_opts='${CFG_OPTS:-}' job=${SLURM_JOB_ID:-} node=$(hostname) start=$STARTED"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
test -f $RUN_DIR/run.json && { echo "REFUSING: $RUN_DIR already finalized"; exit 1; }
mkdir -p $RUN_DIR
# freeze the inputs of this run (the repo copy may be re-synced while the job runs)
[ -f $RUN_DIR/config_src.py ] || cp $REPO/$CONFIG $RUN_DIR/config_src.py
[ -f $RUN_DIR/git_info.json ] || cp $REPO/GIT_INFO_mmpose.json $RUN_DIR/git_info.json
echo "${CFG_OPTS:-}" > $RUN_DIR/cfg_opts.txt
cd $R/mmpose            # mmpose v1.3.2 source tree (tools/train.py)
RESUME=""
if [ -f $RUN_DIR/last_checkpoint ]; then RESUME="--resume"; echo "resuming: $(cat $RUN_DIR/last_checkpoint)"; fi
python tools/train.py $REPO/$CONFIG --work-dir $RUN_DIR $RESUME \
  --cfg-options randomness.seed=$SEED ${CFG_OPTS:-}

# fully-resolved config dumped by mmengine at train start -> used for every later evaluation
cp $RUN_DIR/$(basename $CONFIG) $RUN_DIR/config.py
BEST=$(ls $RUN_DIR/best_coco_AP_epoch_*.pth | tail -1)
echo "selected (best Val400 AP): $BEST"
python $REPO/mmpose_exps/tools/eval_ckpt.py $RUN_DIR/config.py $BEST --split val400 --out-dir $RUN_DIR
python $REPO/eval/eval_coco.py --gt $R/data/splits/val400.json --pred $RUN_DIR/preds_val400.json \
  --out $RUN_DIR/metrics_val400.json --per-image $RUN_DIR/per_image_val400.json --run-id $RUN_ID
python $REPO/mmpose_exps/tools/finalize_run.py $RUN_DIR --run-id $RUN_ID \
  --config $RUN_DIR/config_src.py --config-name $(basename $CONFIG) --seed $SEED --started $STARTED
# keep only the selected checkpoint + the resumable last one (epoch_*.pth is <= 1 by max_keep_ckpts)
echo "done $(date -Is)"
