#!/usr/bin/env bash
# One FiDIP (DarkPose HRNet-W48 384x288) training run with the upstream script, then evaluation.
#   bash scripts/train_fidip.sh RUN_ID SEED LAMBDA [EVAL_SETS] [extra yacs opts ...]
# Recipe = experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml (as shipped: Adam lr 1e-4, 20 epochs,
# batch 20, idx<=108 frozen, balanced real/synthetic batches) plus
#   TRAIN.CHECKPOINT  models/coco/w48_384x288.pth   (upstream README omits it -> would train from random init)
#   TRAIN.PRE_EPOCH 10 with the shipped train_pre_infant set (domain-classifier pre-training, as _custom.yaml)
#   TRAIN.LAMBDA $LAMBDA                            (yaml ships 0.0, i.e. plain fine-tuning)
#   DATASET.TEST_SET val400                         (per-epoch validation + model_best on Val400, not Test100)
#   SEED, WORKERS 8, DEBUG.DEBUG False              (seeded repeats; speed/logging only)
# EVAL_SETS (default "test100 val400"; "none" to skip; "val400" for Val400-only runs) are evaluated for both
# checkpoints: best = model_best.pth (best Val400 AP epoch), final = final_state.pth (last epoch).
set -euo pipefail
R=/scratch/purohitham.v/fidip
RUN_ID=$1; SEED=$2; LAMBDA=$3; EVAL_SETS=${4:-"test100 val400"}; shift 3; [[ $# -gt 0 ]] && shift
RUN_DIR=$R/runs/$RUN_ID
UP=$R/repo/third_party/Infant-Pose-Estimation
CFG=experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml
PY=$R/envs/fidip/bin/python
export PYTHONNOUSERSITE=1
if [[ -e $RUN_DIR/output ]]; then echo "ERROR: $RUN_DIR/output exists (AUTO_RESUME would pick up a stale checkpoint)"; exit 1; fi
mkdir -p "$RUN_DIR"
ARGS="tools/train_adaptive_model_hrnet.py --cfg $CFG TRAIN.CHECKPOINT $R/models/coco/w48_384x288.pth TRAIN.PRE_EPOCH 10 TRAIN.LAMBDA $LAMBDA DATASET.ROOT $R/data/syrip/ DATASET.TRAIN_SET train_infant DATASET.TRAIN_PRE_SET train_pre_infant DATASET.TEST_SET val400 OUTPUT_DIR $RUN_DIR/output LOG_DIR $RUN_DIR/log WORKERS 8 DEBUG.DEBUG False SEED $SEED $*"
python3 -I $R/repo/scripts/write_run_json.py --run-dir "$RUN_DIR" --run-id "$RUN_ID" --stamp start \
  --cmd "cd $UP && $PY $ARGS" --ckpt "init_coco=$R/models/coco/w48_384x288.pth" \
  --set "seed=$SEED" --set "lambda=$LAMBDA" --set "cfg=\"$CFG\"" --set "selection=\"model_best.pth = best Val400 AP (DATASET.TEST_SET val400); final_state.pth = last epoch\""
cd "$UP"
t0=$(date +%s)
echo "[train_fidip $(date '+%F %T')] $RUN_ID: $PY $ARGS"
set +e; $PY $ARGS > "$RUN_DIR/train_stdout.log" 2>&1; rc=$?; set -e
python3 -I $R/repo/scripts/write_run_json.py --run-dir "$RUN_DIR" --run-id "$RUN_ID" \
  --set "train_seconds=$(( $(date +%s) - t0 ))" --set "train_exit_code=$rc"
FO=$RUN_DIR/output/syrip/adaptive_pose_hrnet/$(basename $CFG .yaml)
cp "$FO"/*_train.log "$RUN_DIR/train.log" 2>/dev/null || true
[[ -f $FO/epochs.csv ]] && cp "$FO/epochs.csv" "$RUN_DIR/epochs.csv"
[[ $rc == 0 ]] || { echo "ERROR: training exited $rc"; tail -30 "$RUN_DIR/train_stdout.log"; exit $rc; }
ln -sfn "$FO/model_best.pth" "$RUN_DIR/ckpt_best_val400.pth"
ln -sfn "$FO/final_state.pth" "$RUN_DIR/ckpt_final.pth"
best_epoch=$(python3 -I -c "import csv,sys; r=list(csv.DictReader(open(sys.argv[1]))); b=max(r,key=lambda x:(float(x['AP']),-int(x['epoch']))); print(b['epoch'], b['AP'])" "$RUN_DIR/epochs.csv")
python3 -I $R/repo/scripts/write_run_json.py --run-dir "$RUN_DIR" --run-id "$RUN_ID" \
  --ckpt "best_val400=$FO/model_best.pth" --ckpt "final=$FO/final_state.pth" \
  --set "best_val400_epoch_and_AP=\"$best_epoch\""
echo "[train_fidip] best Val400 epoch/AP: $best_epoch"
cd $R/repo
if [[ $EVAL_SETS != none ]]; then
  bash scripts/eval_upstream.sh "$RUN_ID" $CFG "$FO/model_best.pth" best $EVAL_SETS
  bash scripts/eval_upstream.sh "$RUN_ID" $CFG "$FO/final_state.pth" final $EVAL_SETS
fi
rm -f "$FO/checkpoint.pth"   # ~1 GB resume state (optimizer etc.); best/final state dicts are kept
python3 -I $R/repo/scripts/write_run_json.py --run-dir "$RUN_DIR" --run-id "$RUN_ID" --stamp end
echo "[train_fidip $(date '+%F %T')] $RUN_ID done"
