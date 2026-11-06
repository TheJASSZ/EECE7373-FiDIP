#!/usr/bin/env bash
# Evaluate one checkpoint with the upstream FiDIP test script on one or more SyRIP splits.
#   bash scripts/eval_upstream.sh RUN_ID CFG CKPT TAG SET [SET ...]
# RUN_ID : run directory $R/runs/RUN_ID (created)        CFG : yaml path relative to the upstream root
# CKPT   : checkpoint file (state_dict)                  TAG : '' for released models, e.g. 'best'/'final'
# SET    : DATASET.TEST_SET, e.g. test100 val400 validate500 validate_infant (upstream layout, see notes)
# Writes per SET: preds_<SET>[_<TAG>].json, metrics_upstream_<SET>[_<TAG>].txt, eval_<SET>[_<TAG>].log,
# and merges provenance into run.json. Upstream outputs (incl. the large feature.npy) stay under
# $R/runs/RUN_ID/upstream_eval/.
set -euo pipefail
R=/scratch/purohitham.v/fidip
RUN_ID=$1; CFG=$2; CKPT=$3; TAG=$4; shift 4
RUN_DIR=$R/runs/$RUN_ID
UP=$R/repo/third_party/Infant-Pose-Estimation
PY=$R/envs/fidip/bin/python
export PYTHONNOUSERSITE=1
mkdir -p "$RUN_DIR"
sfx=${TAG:+_$TAG}
python3 -I $R/repo/scripts/write_run_json.py --run-dir "$RUN_DIR" --run-id "$RUN_ID" --ckpt "eval${sfx}=$CKPT" --stamp start
cd "$UP"
for SET in "$@"; do
  OUT=$RUN_DIR/upstream_eval/${SET}${sfx}
  ARGS="tools/test_adaptive_model.py --cfg $CFG TEST.MODEL_FILE $CKPT TEST.USE_GT_BBOX True WORKERS 8 DEBUG.DEBUG False DATASET.ROOT $R/data/syrip/ DATASET.TEST_SET $SET OUTPUT_DIR $OUT/output LOG_DIR $OUT/log"
  CMD="$PY $ARGS"
  LOG=$RUN_DIR/eval_${SET}${sfx}.log
  echo "[eval_upstream $(date '+%F %T')] $RUN_ID $SET$sfx: $CMD"
  t0=$(date +%s)
  set +e; $PY $ARGS > "$LOG" 2>&1; rc=$?; set -e
  grep -vE "^(batch|list)$" "$LOG" | tail -25 || true
  [[ $rc == 0 ]] || { echo "ERROR: test script exited $rc for $SET"; exit $rc; }
  res=$(find "$OUT/output" -name "keypoints_${SET}_results_0.json" | head -1)
  [[ -s "$res" ]] || { echo "ERROR: no results json for $SET"; exit 1; }
  cp "$res" "$RUN_DIR/preds_${SET}${sfx}.json"
  { echo "# upstream-printed metrics: $RUN_ID, split $SET${TAG:+, checkpoint $TAG}"
    echo "# ckpt $CKPT"
    echo "# cmd  $CMD"
    grep -E "Average (Precision|Recall)" "$LOG" || true
    grep -E "^\| (Arch|---|adaptive)" "$LOG" | tail -3 || true
  } > "$RUN_DIR/metrics_upstream_${SET}${sfx}.txt"
  cat "$RUN_DIR/metrics_upstream_${SET}${sfx}.txt" | tail -3
  python3 -I $R/repo/scripts/write_run_json.py --run-dir "$RUN_DIR" --run-id "$RUN_ID" --cmd "cd $UP && $CMD" \
    --set "eval_seconds_${SET}${sfx}=$(( $(date +%s) - t0 ))"
done
python3 -I $R/repo/scripts/write_run_json.py --run-dir "$RUN_DIR" --run-id "$RUN_ID" --stamp end
