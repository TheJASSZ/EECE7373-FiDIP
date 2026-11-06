#!/usr/bin/env python
"""ensemble_preds.py -- average keypoint predictions from N prediction JSONs into one.

  python eval/ensemble_preds.py --gt GT.json --preds A.json B.json C.json --out ens.json \
      [--weighting {mean,conf}] [--match {auto,single,gtbox,centroid}]

Instances are matched per image:
- single:   every file has at most one instance per image (SyRIP with GT boxes), so they are averaged directly.
- gtbox:    each instance is assigned to the GT *box* (never GT keypoints, so no label leakage)
            with the largest IoU between the box and the instance's keypoint extent.
- centroid: anchors = the first file's instances; other files are matched greedily by nearest
            keypoint centroid.
- auto:     single if possible, else gtbox when a GT is given, else centroid.

Averaging (per keypoint):
- mean: plain mean of coordinates.
- conf: confidence-weighted mean of coordinates.
- Confidence is the mean confidence across models.
- The formula is x0 + Σ w_i (x_i - x0) / Σ w_i, so an ensemble of identical files is bit-identical to its input.
- `score` = the shared rescoring rule on the averaged confidences (eval_coco.py rescoring recomputes it anyway).

A sidecar <out>.meta.json records the inputs (paths + sha256) and settings.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_coco as ev  # noqa: E402


def _avg(stack, weights):
    """stack: (N, K) values; weights: (N, K) >= 0. Exact for identical rows."""
    x0 = stack[0]
    w = np.asarray(weights, dtype=float)
    wsum = w.sum(axis=0)
    safe = np.where(wsum > 0, wsum, 1.0)
    upd = (w * (stack - x0)).sum(axis=0) / safe
    return np.where(wsum > 0, x0 + upd, x0)


def average_instances(insts, weighting="mean"):
    """insts: list of (K,3) arrays [x, y, conf] -> averaged (K,3)."""
    S = np.stack(insts)                      # (N, K, 3)
    conf = S[:, :, 2]
    if weighting == "conf":
        w = np.clip(conf, 0, None)
        w = np.where(w.sum(axis=0, keepdims=True) > 0, w, 1.0)
    elif weighting == "mean":
        w = np.ones_like(conf)
    else:
        raise ValueError(weighting)
    x = _avg(S[:, :, 0], w)
    y = _avg(S[:, :, 1], w)
    c = _avg(conf, np.ones_like(conf))
    return np.stack([x, y, c], axis=1)


def _extent_box(k):
    x, y = k[:, 0], k[:, 1]
    return np.array([x.min(), y.min(), x.max() - x.min(), y.max() - y.min()])


def _iou(a, b):
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u > 0 else 0.0


def ensemble(G, pred_files, weighting="mean", match="auto"):
    per_file = []
    infos = []
    for f in pred_files:
        preds, info, _ = ev.load_predictions(f, G)
        by_img = defaultdict(list)
        for p in preds:
            by_img[p["image_id"]].append(p["kpts"])
        per_file.append(by_img)
        infos.append(info)
    img_ids = sorted(set().union(*[set(d) for d in per_file]))
    max_per_img = max((len(d[i]) for d in per_file for i in d), default=0)
    mode = match
    if match == "auto":
        mode = "single" if max_per_img <= 1 else ("gtbox" if G is not None else "centroid")
    if mode == "single" and max_per_img > 1:
        raise ValueError("--match single but some file has several instances per image")
    out, n_partial = [], 0
    for img in img_ids:
        groups = []
        if mode == "single":
            g = [d[img][0] for d in per_file if d.get(img)]
            groups.append(g)
        elif mode == "gtbox":
            boxes = [a["bbox"] for a in G.anns_by_img.get(img, [])]
            buckets = defaultdict(list)
            for d in per_file:
                taken = set()
                for k in d.get(img, []):
                    ious = [(_iou(_extent_box(k), b), j) for j, b in enumerate(boxes) if j not in taken]
                    if not ious:
                        continue
                    best = max(ious)[1]
                    taken.add(best)
                    buckets[best].append(k)
            groups = [buckets[j] for j in sorted(buckets)]
        elif mode == "centroid":
            anchors = per_file[0].get(img, [])
            groups = [[a] for a in anchors]
            for d in per_file[1:]:
                cands = list(d.get(img, []))
                for gi, a in enumerate(anchors):
                    if not cands:
                        break
                    c0 = a[:, :2].mean(0)
                    j = int(np.argmin([np.linalg.norm(k[:, :2].mean(0) - c0) for k in cands]))
                    groups[gi].append(cands.pop(j))
        else:
            raise ValueError(mode)
        for g in groups:
            if not g:
                continue
            if len(g) < len(per_file):
                n_partial += 1
            k = average_instances(g, weighting)
            out.append({"image_id": int(img), "category_id": G.cat_id if G else 1,
                        "keypoints": [float(v) for v in k.reshape(-1)],
                        "score": ev.rescore_value(k[:, 2]), "n_models": len(g)})
    meta = OrderedDict(tool="eval/ensemble_preds.py", weighting=weighting, match=mode,
                       inputs=[OrderedDict(path=i["path"], sha256=i["sha256"], n=i["n_resolved"],
                                           run_id=ev.infer_run_id(i["path"])) for i in infos],
                       n_out=len(out), n_instances_missing_some_model=n_partial)
    return out, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="GT json (image-id resolution; boxes for --match gtbox)")
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weighting", choices=["mean", "conf"], default="mean")
    ap.add_argument("--match", choices=["auto", "single", "gtbox", "centroid"], default="auto")
    ap.add_argument("--area", choices=["gt", "bbox"], default="gt")
    a = ap.parse_args(argv)
    G = ev.GroundTruth(a.gt, area_rule=a.area)
    out, meta = ensemble(G, a.preds, a.weighting, a.match)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f)
    with open(os.path.splitext(a.out)[0] + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[ensemble] {len(a.preds)} files -> {len(out)} instances ({meta['match']}, {a.weighting}) -> {a.out}")
    if meta["n_instances_missing_some_model"]:
        print(f"  WARNING: {meta['n_instances_missing_some_model']} instances lack some model's prediction")
    return 0


if __name__ == "__main__":
    sys.exit(main())
