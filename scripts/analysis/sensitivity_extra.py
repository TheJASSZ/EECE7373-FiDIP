#!/usr/bin/env python
"""Extra evaluation-condition sensitivities for the released checkpoint (tab_sensitivity):
  (a) Val400 AP with the shipped constant placeholder area (raw validate500 annotations, Val400 ids), vs area=w*h;
  (b) MMPose CocoMetric backend (xtcocotools) on the same Test100 predictions, if xtcocotools is importable.
Writes $R/analysis/tables/sensitivity_extra_<RUN_ID>.json."""
import contextlib, copy, io, json, os, sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "eval"))
import eval_coco as ev
R = os.environ.get("R", "/scratch/purohitham.v/fidip")
rid = sys.argv[1]
OUT = f"{R}/analysis/tables/sensitivity_extra_{rid}.json"
out = json.load(open(OUT)) if os.path.exists(OUT) else {}      # merge: the two envs add different keys
out["run_id"] = rid
raw = json.load(open(f"{R}/data/raw/SyRIP/annotations/validate500/person_keypoints_validate_infant.json"))
v400 = json.load(open(f"{R}/data/splits/val400.json"))
ids = {im["id"] for im in v400["images"]}
const = dict(raw, images=[im for im in raw["images"] if im["id"] in ids],
             annotations=[a for a in raw["annotations"] if a["image_id"] in ids])
pv = f"{R}/runs/{rid}/preds_val400.json"
if os.path.exists(pv):
    o1, _ = ev.evaluate(const, pv, allow_constant_area=True, with_per_image=False)
    o2, _ = ev.evaluate(copy.deepcopy(v400), pv, with_per_image=False)
    out["val400_AP_const_area"] = o1["metrics"]["AP"]
    out["val400_AP_wh_area"] = o2["metrics"]["AP"]
    out["val400_const_area_value"] = const["annotations"][0]["area"]
pt = f"{R}/runs/{rid}/preds_test100.json"
try:
    from xtcocotools.coco import COCO
    from xtcocotools.cocoeval import COCOeval
    G = ev.GroundTruth(f"{R}/data/splits/test100.json")
    preds, _, _ = ev.load_predictions(pt, G)
    res = ev._to_coco_results(ev.postprocess(preds))
    with contextlib.redirect_stdout(io.StringIO()):
        c = COCO(f"{R}/data/splits/test100.json")
        d = c.loadRes(copy.deepcopy(res))
        E = COCOeval(c, d, "keypoints"); E.evaluate(); E.accumulate(); E.summarize()
    o, _ = ev.evaluate(f"{R}/data/splits/test100.json", pt, with_per_image=False)
    out.update(test100_AP_xtcocotools=float(E.stats[0]), test100_AP_eval_coco_paper=o["metrics"]["AP"],
               test100_AP_eval_coco_fixed=o["metrics_fixed"]["AP"])
except ImportError:
    pass                                                        # analysis env: no xtcocotools
json.dump(out, open(OUT, "w"), indent=1)
print(out)
