#!/usr/bin/env bash
# IMPROVE stream: the body of ONE fine-tuning run (called by scripts/slurm/train_mmpose.sbatch and
# scripts/slurm/train_pack_mmpose.sbatch; expects the mmpose env to be active).
#   1) tools/train.py (Val400 every epoch, save_best coco/AP), resuming from last_checkpoint if any
#   2) the selected checkpoint (best Val400 epoch) re-run on Val400 -> preds_val400.json
#   3) eval/eval_coco.py on it                                  -> metrics_val400.json
#   4) finalize_run.py                                         -> run.json, val_curve.json, registry_row.csv
# Never touches Test100.
# usage: run_one_mmpose.sh RUN_ID CONFIG(repo-relative) SEED "CFG_OPTS"
set -euo pipefail
RUN_ID=$1; CONFIG=$2; SEED=$3; CFG_OPTS=${4:-}
R=/scratch/purohitham.v/fidip
REPO=${REPO:-$R/repo}   # sbatch exports REPO=$R/snapshots/<job> (D19)
RUN_DIR=$R/runs/$RUN_ID
export PYTHONHASHSEED=$SEED
STARTED=$(date -Is)
echo "run=$RUN_ID config=$CONFIG seed=$SEED cfg_opts='$CFG_OPTS' job=${SLURM_JOB_ID:-} node=$(hostname) start=$STARTED"
test -f $RUN_DIR/run.json && { echo "REFUSING: $RUN_DIR already finalized"; exit 1; }
mkdir -p $RUN_DIR
# freeze the inputs of this run (the repo copy may be re-synced while the job runs)
[ -f $RUN_DIR/config_src.py ] || cp $REPO/$CONFIG $RUN_DIR/config_src.py
[ -f $RUN_DIR/git_info.json ] || cp $REPO/GIT_INFO_mmpose.json $RUN_DIR/git_info.json
echo "$CFG_OPTS" > $RUN_DIR/cfg_opts.txt
cd $R/mmpose            # mmpose v1.3.2 source tree (tools/train.py)
RESUME=""
if [ -f $RUN_DIR/last_checkpoint ]; then RESUME="--resume"; echo "resuming: $(cat $RUN_DIR/last_checkpoint)"; fi
python tools/train.py $REPO/$CONFIG --work-dir $RUN_DIR $RESUME \
  --cfg-options randomness.seed=$SEED $CFG_OPTS
# fully-resolved config dumped by mmengine at train start -> used for every later evaluation
cp $RUN_DIR/$(basename $CONFIG) $RUN_DIR/config.py
BEST=$(ls $RUN_DIR/best_coco_AP_epoch_*.pth | tail -1)
echo "selected (best Val400 AP): $BEST"
python $REPO/mmpose_exps/tools/eval_ckpt.py $RUN_DIR/config.py $BEST --split val400 --out-dir $RUN_DIR --run-id $RUN_ID
python $REPO/eval/eval_coco.py --gt $R/data/splits/val400.json --pred $RUN_DIR/preds_val400.json \
  --out $RUN_DIR/metrics_val400.json --per-image $RUN_DIR/per_image_val400.json --run-id $RUN_ID
python $REPO/mmpose_exps/tools/finalize_run.py $RUN_DIR --run-id $RUN_ID \
  --config $RUN_DIR/config_src.py --config-name $(basename $CONFIG) --seed $SEED --started $STARTED
# finalized: drop the resumable optimizer-state checkpoint (only needed for --resume); keep best_*.pth
rm -f $RUN_DIR/epoch_*.pth $RUN_DIR/last_checkpoint
echo "done $RUN_ID $(date -Is)"
