#!/usr/bin/env python
"""Empirical check of the annotation-id-0 claim on the REAL Test100 GT. Noisy GT-derived
predictions (several noise levels and score orders) are scored with:
  (1) eval/eval_coco.py, paper and fixed protocols;
  (2) raw pycocotools COCOeval (whatever version is installed);
  (3) raw xtcocotools COCOeval (MMPose 1.3.2's CocoMetric backend), if installed.
Expected: pycocotools == paper, xtcocotools == fixed. Run in $R/envs/mmpose (it has all three)."""
import contextlib
import copy
import io
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "eval"))
import eval_coco as ev  # noqa: E402

gt_path = sys.argv[1]
gt = json.load(open(gt_path))


def raw_eval(lib, gtd, dets):
    if lib == "pycocotools":
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    else:
        from xtcocotools.coco import COCO
        from xtcocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        c = COCO()
        c.dataset = copy.deepcopy(gtd)
        c.createIndex()
        d = c.loadRes(copy.deepcopy(dets))
        E = COCOeval(c, d, "keypoints")
        E.evaluate(); E.accumulate(); E.summarize()
    return E.stats[[0, 1, 2, 5, 6, 7]]


rng = np.random.default_rng(0)
rows = []
for noise in (0.0, 0.01, 0.02, 0.04):
    for order in ("conf", "id0_first", "id0_last"):
        dets = []
        z = rng.standard_normal((len(gt["annotations"]), 17, 2))
        for j, a in enumerate(gt["annotations"]):
            g = np.asarray(a["keypoints"], float).reshape(17, 3)
            s = np.sqrt(a["bbox"][2] * a["bbox"][3])
            xy = g[:, :2] + noise * s * z[j]
            c = np.clip(0.9 - 3 * noise + 0.05 * rng.standard_normal(17), 0.25, 1)
            if order == "id0_first":
                c = np.full(17, 0.99 if a["image_id"] == 0 else 0.5 + 0.004 * (a["image_id"] % 100))
            if order == "id0_last":
                c = np.full(17, 0.21 if a["image_id"] == 0 else 0.5 + 0.004 * (a["image_id"] % 100))
            kp = np.concatenate([xy, c[:, None]], 1).reshape(-1).tolist()
            dets.append({"image_id": a["image_id"], "category_id": 1, "keypoints": kp, "score": ev.rescore_value(c)})
        out, _ = ev.evaluate(copy.deepcopy(gt), copy.deepcopy(dets), with_per_image=False)
        r = {"noise": noise, "order": order,
             "ours_paper": [out["metrics"][k] for k in ("AP", "AP50", "AP75", "AR", "AR50", "AR75")],
             "ours_fixed": [out["metrics_fixed"][k] for k in ("AP", "AP50", "AP75", "AR", "AR50", "AR75")]}
        for lib in ("pycocotools", "xtcocotools"):
            try:
                r[lib] = raw_eval(lib, gt, dets).tolist()
            except ImportError:
                r[lib] = None
        rows.append(r)

ok_py = ok_xt = True
for r in rows:
    py, xt = r["pycocotools"], r["xtcocotools"]
    d_py = max(abs(a - b) for a, b in zip(py, r["ours_paper"])) if py else None
    d_xt = max(abs(a - b) for a, b in zip(xt, r["ours_fixed"])) if xt else None
    ok_py &= (d_py is None or d_py < 1e-12)
    ok_xt &= (d_xt is None or d_xt < 1e-12)
    print(f"noise={r['noise']:.2f} {r['order']:10s} AP paper {100*r['ours_paper'][0]:6.2f} fixed {100*r['ours_fixed'][0]:6.2f}"
          f" | pycocotools {100*py[0] if py else float('nan'):6.2f} (max|d| vs paper {d_py})"
          f" | xtcocotools {100*xt[0] if xt else float('nan'):6.2f} (max|d| vs fixed {d_xt})")
from importlib.metadata import version
vers = {}
for lib in ("pycocotools", "xtcocotools", "numpy"):
    try:
        vers[lib] = version(lib)
    except Exception:
        vers[lib] = None
print("versions", vers)
print("RESULT pycocotools==paper:", ok_py, " xtcocotools==fixed:", ok_xt)
json.dump({"rows": rows, "versions": vers, "pycocotools_eq_paper": ok_py, "xtcocotools_eq_fixed": ok_xt},
          open(sys.argv[2], "w"), indent=1)
