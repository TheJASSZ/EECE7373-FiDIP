#!/usr/bin/env bash
# First full analysis of the released FiDIP checkpoint(s). Run inside sbatch:
#   sbatch scripts/slurm/analysis.sbatch script scripts/analysis/run_released.sh [RUN_ID]
set -euo pipefail
R=${R:-/scratch/purohitham.v/fidip}
A=$R/analysis
RID=${1:-R001_fidip-w48_released}
LABEL=${LABEL:-"FiDIP (released)"}
SLOT=${SLOT:-1}
IMG=$R/data/raw/SyRIP/images/validate_infant
POST=$R/data/raw/syrip_for_train/syrip/annotations/person_keypoints_validate_infant.json
E=$A/eval/$RID
mkdir -p $A/manifests $A/figures $A/tables $A/bootstrap

python scripts/analysis_pipeline.py eval-all
python scripts/analysis/crosscheck_upstream.py || echo "WARNING: upstream cross-check mismatch"
for s in test100 val400; do
  test -f $E/per_image_$s.json || { echo "missing $E/per_image_$s.json"; exit 3; }
done

python - "$RID" "$LABEL" "$E" "$POST" "$A/manifests" "$SLOT" <<'PY'
import json, sys
rid, label, E, post, out, slot = sys.argv[1:]
for split in ("test100", "val400"):
    m = {"split": split, "gt": f"/scratch/purohitham.v/fidip/data/splits/{split}.json",
         "models": [{"key": "released", "label": label, "slot": int(slot),
                     "runs": [{"run_id": rid, "per_image": f"{E}/per_image_{split}.json",
                               "metrics": f"{E}/metrics_{split}.json"}]}],
         "focus": ["released"],
         "published": {"label": "FiDIP (paper)", "AP": 0.936, "AP50": 0.985, "AP75": 0.985, "AR": 0.946}}
    if split == "test100":
        m["posture"] = post
        import os
        vp = "/scratch/purohitham.v/fidip/analysis/splits_derived/README.json"
        if os.path.exists(vp):
            m["leak_ids"] = json.load(open(vp))["test100_clean"]["excluded_test_ids"]
    json.dump(m, open(f"{out}/{rid}_{split}.json", "w"), indent=1)
PY

for s in test100 val400; do
  python eval/analyze.py models --manifest $A/manifests/${RID}_$s.json --outdir $A/figures/${RID}/$s --tables $A/tables/${RID}_$s
  python eval/visualize.py gallery  --per-image $E/per_image_$s.json --images $IMG --label "$LABEL" --slot $SLOT \
         --out $A/figures/${RID}/$s/gallery_${RID}_$s
  python eval/visualize.py failures --per-image $E/per_image_$s.json --images $IMG --label "$LABEL" --slot $SLOT \
         --out $A/figures/${RID}/$s/failures_${RID}_$s
done
python eval/analyze.py splits --label "$LABEL" --name $RID --a test100=$E/per_image_test100.json \
       --b val400=$E/per_image_val400.json --outdir $A/figures/${RID}/splits --tables $A/tables/${RID}_splits
python eval/bootstrap_ci.py single --gt $R/data/splits/test100.json --a $R/runs/$RID/preds_test100.json \
       --out $A/bootstrap/bootstrap_single_${RID}_test100.json
python eval/bootstrap_ci.py single --gt $R/data/splits/val400.json --a $R/runs/$RID/preds_val400.json \
       --out $A/bootstrap/bootstrap_single_${RID}_val400.json
# D13: paper-protocol AP of these exact predictions as test0's confidence rank moves
python eval/analyze.py id0rank --gt $R/data/splits/test100.json --model "$LABEL=$R/runs/$RID/preds_test100.json:$SLOT" \
       --name $RID --outdir $A/figures/${RID}/test100 --tables $A/tables/${RID}_test100
# Test100-clean (D16) metrics come from eval-all; defective-test11 impact: test100 vs validate_infant per image
if [ -f $E/per_image_validate_infant.json ]; then
python - "$E" "$A/tables/${RID}_test100/defect_test11.csv" <<'PY'
import json, sys, csv
E, out = sys.argv[1:]
a = {r["image_id"]: r for r in json.load(open(f"{E}/per_image_test100.json"))["records"]}
b = {r["image_id"]: r for r in json.load(open(f"{E}/per_image_validate_infant.json"))["records"]}
ma, mb = json.load(open(f"{E}/metrics_test100.json")), json.load(open(f"{E}/metrics_validate_infant.json"))
diff = sorted(((b[i]["oks"] - a[i]["oks"], i) for i in a), key=lambda t: t[0])[:3]
rows = [{"image_id": i, "oks_test100_imgs": a[i]["oks"], "oks_validate_infant_imgs": b[i]["oks"], "delta": d} for d, i in diff]
rows.append({"image_id": "AP_paper", "oks_test100_imgs": ma["metrics"]["AP"], "oks_validate_infant_imgs": mb["metrics"]["AP"],
             "delta": mb["metrics"]["AP"] - ma["metrics"]["AP"]})
rows.append({"image_id": "AP_fixed", "oks_test100_imgs": ma["metrics_fixed"]["AP"],
             "oks_validate_infant_imgs": mb["metrics_fixed"]["AP"], "delta": mb["metrics_fixed"]["AP"] - ma["metrics_fixed"]["AP"]})
w = csv.DictWriter(open(out, "w", newline=""), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("defect:", rows)
PY
fi
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
