#!/usr/bin/env bash
# Score every preds_<split>[_<tag>].json in the given run dirs with the project evaluator eval/eval_coco.py
# (both protocols: paper = GT as shipped, fixed = GT ids shifted off 0) -> metrics_<split>[_<tag>].json.
# GT: $R/data/splits/<split>.json; validate_infant preds (upstream README layout) are scored against test100.json.
#   bash scripts/eval_coco_repro.sh $R/runs/R010_* ...      (CPU; analysis env)
set -euo pipefail
R=/scratch/purohitham.v/fidip
PY=$R/envs/analysis/bin/python
export PYTHONNOUSERSITE=1
for d in "$@"; do
  rid=$(basename "$d")
  for p in "$d"/preds_*.json; do
    [[ -e $p ]] || continue
    b=$(basename "$p" .json); rest=${b#preds_}; split=${rest%%_best}; split=${split%%_final}
    case $split in validate_infant) gt=$R/data/splits/test100.json ;; *) gt=$R/data/splits/$split.json ;; esac
    out=$d/metrics_${rest}.json
    $PY -I $R/repo/eval/eval_coco.py --gt "$gt" --pred "$p" --out "$out" --run-id "$rid" > "$d/eval_coco_${rest}.log" 2>&1 \
      || { echo "eval_coco FAILED for $p"; tail -n 5 "$d/eval_coco_${rest}.log"; exit 1; }
    echo "$rid $rest: $(tail -n 2 "$d/eval_coco_${rest}.log" | tr '\n' ' ' | cut -c1-230)"
  done
done
