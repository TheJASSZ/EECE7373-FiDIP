#!/usr/bin/env bash
# First full analysis of the released FiDIP checkpoint(s). Run inside sbatch:
#   sbatch scripts/slurm/analysis.sbatch script scripts/analysis/run_released.sh [RUN_ID]
set -euo pipefail
R=${R:-/scratch/purohitham.v/fidip}
A=$R/analysis
RID=${1:-R001_fidip-w48_released}
LABEL=${LABEL:-"FiDIP (released)"}
IMG=$R/data/raw/SyRIP/images/validate_infant
POST=$R/data/raw/syrip_for_train/syrip/annotations/person_keypoints_validate_infant.json
E=$A/eval/$RID
mkdir -p $A/manifests $A/figures $A/tables $A/bootstrap

python scripts/analysis_pipeline.py eval-all
for s in test100 val400; do
  test -f $E/per_image_$s.json || { echo "missing $E/per_image_$s.json"; exit 3; }
done

python - "$RID" "$LABEL" "$E" "$POST" "$A/manifests" <<'PY'
import json, sys
rid, label, E, post, out = sys.argv[1:]
for split in ("test100", "val400"):
    m = {"split": split, "gt": f"/scratch/purohitham.v/fidip/data/splits/{split}.json",
         "models": [{"key": "released", "label": label, "slot": 1,
                     "runs": [{"run_id": rid, "per_image": f"{E}/per_image_{split}.json",
                               "metrics": f"{E}/metrics_{split}.json"}]}],
         "focus": ["released"],
         "published": {"label": "FiDIP (paper)", "AP": 0.936, "AP50": 0.985, "AP75": 0.985, "AR": 0.946}}
    if split == "test100":
        m["posture"] = post
    json.dump(m, open(f"{out}/{rid}_{split}.json", "w"), indent=1)
PY

for s in test100 val400; do
  python eval/analyze.py models --manifest $A/manifests/${RID}_$s.json --outdir $A/figures/${RID}/$s --tables $A/tables/${RID}_$s
  python eval/visualize.py gallery  --per-image $E/per_image_$s.json --images $IMG --label "$LABEL" --slot 1 \
         --out $A/figures/${RID}/$s/gallery_${RID}_$s
  python eval/visualize.py failures --per-image $E/per_image_$s.json --images $IMG --label "$LABEL" --slot 1 \
         --out $A/figures/${RID}/$s/failures_${RID}_$s
done
python eval/analyze.py splits --label "$LABEL" --name $RID --a test100=$E/per_image_test100.json \
       --b val400=$E/per_image_val400.json --outdir $A/figures/${RID}/splits --tables $A/tables/${RID}_splits
python eval/bootstrap_ci.py single --gt $R/data/splits/test100.json --a $R/runs/$RID/preds_test100.json \
       --out $A/bootstrap/bootstrap_single_${RID}_test100.json
python eval/bootstrap_ci.py single --gt $R/data/splits/val400.json --a $R/runs/$RID/preds_val400.json \
       --out $A/bootstrap/bootstrap_single_${RID}_val400.json
# v100-vs-v500 GT-box 2x2 (needs preds_validate500.json made with validate500 boxes)
if [ -f $R/runs/$RID/preds_validate500.json ]; then
  python eval/analyze.py gtbox --pred-v100box $R/runs/$RID/preds_test100.json \
         --pred-v500box $R/runs/$RID/preds_validate500.json --gt-v100 $R/data/splits/test100.json \
         --gt-v500 $R/data/splits/validate500.json --tables $A/tables/${RID}_gtbox
else
  python eval/analyze.py gtbox --pred-v100box $R/runs/$RID/preds_test100.json --gt-v100 $R/data/splits/test100.json \
         --gt-v500 $R/data/splits/validate500.json --tables $A/tables/${RID}_gtbox
fi
echo "done $RID"
