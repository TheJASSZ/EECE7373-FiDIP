#!/usr/bin/env python
"""eval_coco.py -- THE evaluator for every reported keypoint number in this project.

One pycocotools COCOeval (iouType='keypoints', default COCO-17 sigmas, GT boxes,
GT `area` as the OKS scale, maxDets=20) applied to a COCO-style prediction list, after
one shared post-processing rule:

  1. Rescoring (default, ``--rescore``): every instance score is recomputed as
     ``mean(keypoint confidences > 0.2) * box_score`` with ``box_score = 1`` (GT boxes).
     This is exactly upstream FiDIP's rule (lib/dataset/syrip.py::evaluate) and MMPose's
     ``score_mode='bbox_keypoint'``, so predictions from different frameworks are ranked
     identically. Un-rescored numbers (the file's own ``score``) are always reported too.
  2. OKS-NMS at 0.9 per image (upstream lib/nms/nms.py::oks_nms, replicated exactly;
     a no-op when there is one GT box per image, as in SyRIP).
  3. COCOeval.

Two matching protocols are always computed (see eval/README.md):

* ``paper`` (primary, ``metrics``): the GT exactly as distributed. SyRIP Test100 contains an
  annotation with ``id == 0`` (test0.jpg); pycocotools stores matched GT ids in
  ``dtMatches`` and treats 0 as "unmatched", so that GT can never be a true positive and
  its detection is always a false positive. This caps AR at 0.99 and is baked into the
  published FiDIP numbers, so this protocol is the one comparable with the paper.
* ``fixed`` (``metrics_fixed``): identical, except GT annotation ids are shifted so none
  is 0. Identical to ``paper`` whenever no annotation id is 0 (e.g. Val400).

Also reported: per-image OKS (best OKS of any detection for each GT instance), and
per-keypoint error normalised by sqrt(GT bbox w*h) with PCK@{0.05,0.1,0.2} under the
same normaliser (for SyRIP Test100 this equals sqrt(area) used by OKS).

Usage:
  python eval/eval_coco.py --gt GT.json --pred preds.json --out metrics.json \
      [--per-image per_image.json] [--no-rescore] [--area {gt,bbox}] [--run-id R001_...]
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
import os
import re
import sys
import warnings
from collections import OrderedDict, defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

EVALUATOR_VERSION = "1.0"

COCO_KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
# Default COCO-17 OKS sigmas (identical to pycocotools Params.kpt_oks_sigmas and upstream nms.py).
COCO_SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62,
                        1.07, 1.07, .87, .87, .89, .89]) / 10.0
# 0-based left/right pairs (upstream SYRIPDataset.flip_pairs).
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
# 0-based skeleton (COCO 1-based skeleton from the SyRIP GT, minus 1).
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
            (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
            (3, 5), (4, 6)]
METRIC_NAMES = ["AP", "AP50", "AP75", "APm", "APl", "AR", "AR50", "AR75", "ARm", "ARl"]
PCK_THRESHOLDS = (0.05, 0.1, 0.2)
DEFAULT_KPT_THR = 0.2   # upstream TEST.IN_VIS_THRE; MMPose keypoint_score_thr
DEFAULT_NMS_THR = 0.9   # upstream TEST.OKS_THRE; MMPose nms_thr
PROTOCOLS = ("paper", "fixed")


# ----------------------------------------------------------------------------- utilities
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(obj):
    """Return (data, path) for a path or an already-loaded object."""
    if isinstance(obj, (str, os.PathLike)):
        with open(obj) as f:
            return json.load(f), str(obj)
    return obj, None


@contextlib.contextmanager
def _quiet(enabled=True):
    """Silence pycocotools' prints (it writes progress lines to stdout)."""
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _coco_from_dataset(dataset):
    coco = COCO()
    coco.dataset = dataset
    with _quiet():
        coco.createIndex()
    return coco


def _none_if_neg(x):
    x = float(x)
    return None if x < 0 else x


def _versions():
    try:
        from importlib.metadata import version
        pcv = version("pycocotools")
    except Exception:  # pragma: no cover
        pcv = "unknown"
    return {"pycocotools": pcv, "numpy": np.__version__, "python": sys.version.split()[0]}


def infer_run_id(path):
    """RUN_ID from a path like .../runs/R001_fidip-w48_released/preds_test100.json."""
    if not path:
        return None
    m = re.search(r"(?:^|/)(R\d{3}[A-Za-z0-9_.\-]*)/", str(path))
    return m.group(1) if m else None


def infer_split(path):
    if not path:
        return None
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    for s in ("test100", "val400", "validate500", "validate100", "train_infant"):
        if s in stem:
            return s
    return stem


# ----------------------------------------------------------------------------- ground truth
class GroundTruth:
    """A validated GT dataset plus the derived per-protocol COCO objects."""

    def __init__(self, gt, area_rule="gt", allow_constant_area=False):
        data, path = _load_json(gt)
        self.path = path
        self.sha256 = sha256_file(path) if path else None
        self.warnings = []
        ds = copy.deepcopy(data)
        for key in ("images", "annotations", "categories"):
            if key not in ds:
                raise ValueError(f"GT is missing '{key}'")
        cats = ds["categories"]
        person = [c for c in cats if c.get("name") == "person"] or cats
        if len(person) != 1:
            raise ValueError(f"expected exactly one (person) category, got {cats}")
        self.cat_id = int(person[0]["id"])
        self.kpt_names = person[0].get("keypoints") or COCO_KPT_NAMES
        self.num_kpts = len(self.kpt_names)
        if self.num_kpts != len(COCO_SIGMAS):
            raise ValueError(f"GT has {self.num_kpts} keypoints; this evaluator assumes COCO-17")

        anns = ds["annotations"]
        n_filled = 0
        for a in anns:
            a["image_id"] = int(a["image_id"])
            a["id"] = int(a["id"])
            kp = np.asarray(a["keypoints"], dtype=float).reshape(-1, 3)
            if kp.shape[0] != self.num_kpts:
                raise ValueError(f"GT ann {a['id']} has {kp.shape[0]} keypoints")
            if "num_keypoints" not in a:
                a["num_keypoints"] = int((kp[:, 2] > 0).sum())
                n_filled += 1
            a.setdefault("iscrowd", 0)
        if n_filled:
            self.warnings.append(f"filled missing num_keypoints for {n_filled} GT annotations")
        ids = [a["id"] for a in anns]
        if len(set(ids)) != len(ids):
            raise ValueError("GT annotation ids are not unique")

        # ---- OKS scale (area) rule + degenerate-area guard
        areas_field = np.array([float(a.get("area", np.nan)) for a in anns])
        bbox_wh = np.array([float(a["bbox"][2]) * float(a["bbox"][3]) for a in anns])
        self.area_rule = area_rule
        if area_rule == "gt":
            if np.isnan(areas_field).any():
                raise ValueError("some GT annotations have no 'area'; pass --area bbox")
            if len(anns) >= 5 and np.ptp(areas_field) == 0 and not allow_constant_area:
                raise ValueError(
                    f"GT 'area' is identical ({areas_field[0]:.6g}) for all {len(anns)} annotations: "
                    "a placeholder, not an object scale (SyRIP validate500/train ship like this). "
                    "OKS would be meaningless. Fix the GT (area = bbox w*h) or pass --area bbox.")
        elif area_rule == "bbox":
            for a, ar in zip(anns, bbox_wh):
                a["area"] = float(ar)
        else:
            raise ValueError(f"unknown area rule {area_rule!r}")
        areas_used = np.array([float(a["area"]) for a in anns])
        if (areas_used <= 0).any():
            raise ValueError("non-positive GT area")
        rel = np.abs(areas_used - bbox_wh) / np.maximum(bbox_wh, 1e-12)
        self.area_check = {
            "rule": area_rule,
            "sqrt_area_min": float(np.sqrt(areas_used.min())),
            "sqrt_area_median": float(np.sqrt(np.median(areas_used))),
            "sqrt_area_max": float(np.sqrt(areas_used.max())),
            "frac_area_equals_bbox_wh": float(np.mean(rel < 1e-6)),
            "n_medium": int(((areas_used >= 32 ** 2) & (areas_used < 96 ** 2)).sum()),
            "n_large": int((areas_used >= 96 ** 2).sum()),
        }

        self.dataset = ds
        self.images = {int(im["id"]): im for im in ds["images"]}
        for im in ds["images"]:
            im["id"] = int(im["id"])
        self.img_ids = sorted(self.images)
        self.anns_by_img = defaultdict(list)
        for a in anns:
            self.anns_by_img[a["image_id"]].append(a)
        self.has_ann_id_0 = 0 in ids
        if self.has_ann_id_0:
            zero = [a for a in anns if a["id"] == 0][0]
            self.warnings.append(
                f"GT contains annotation id 0 (image_id {zero['image_id']}): under pycocotools it can "
                "never be matched ('paper' protocol, comparable with published numbers); "
                "see metrics_fixed for the corrected protocol")
        self.min_ann_id = min(ids) if ids else 1

        # ---- image-id lookup for predictions (int, '27', 'test27', 'test27.jpg', paths)
        self._name_map = {}
        for im in ds["images"]:
            for key in ("file_name", "original_file_name"):
                fn = im.get(key)
                if fn:
                    base = os.path.basename(str(fn))
                    self._name_map[base] = im["id"]
                    self._name_map[os.path.splitext(base)[0]] = im["id"]
        self._coco = {}

    def coco(self, protocol="paper"):
        """COCO object for a protocol ('paper' = ids as given; 'fixed' = no annotation id 0)."""
        if protocol not in PROTOCOLS:
            raise ValueError(protocol)
        if protocol not in self._coco:
            ds = self.dataset
            if protocol == "fixed" and self.min_ann_id < 1:
                ds = copy.deepcopy(ds)
                off = 1 - self.min_ann_id
                for a in ds["annotations"]:
                    a["id"] += off
            self._coco[protocol] = _coco_from_dataset(ds)
        return self._coco[protocol]

    def resolve_image_id(self, value):
        """Map a prediction's image reference to a GT image id, or None."""
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)):
            return None
        if isinstance(value, (int, np.integer)):
            v = int(value)
            return v if v in self.images else None
        if isinstance(value, (float, np.floating)):
            if float(value).is_integer() and int(value) in self.images:
                return int(value)
            return None
        s = str(value).strip()
        if re.fullmatch(r"[+-]?\d+", s):
            v = int(s)
            return v if v in self.images else None
        base = os.path.basename(s)
        for key in (base, os.path.splitext(base)[0]):
            if key in self._name_map:
                return self._name_map[key]
        return None

    def bbox_norm(self, ann):
        """Per-keypoint error normaliser: sqrt(GT bbox w*h)."""
        return math.sqrt(max(float(ann["bbox"][2]) * float(ann["bbox"][3]), 1e-12))


# ----------------------------------------------------------------------------- predictions
def _decode_tricks(v):
    """json_tricks encodes numpy arrays/scalars as {"__ndarray__": data, "dtype": ..., "shape": ...}."""
    if isinstance(v, dict) and "__ndarray__" in v:
        arr = np.asarray(v["__ndarray__"], dtype=v.get("dtype") or float)
        shape = v.get("shape")
        return arr.reshape(shape) if shape is not None and np.prod(shape) == arr.size else arr
    if isinstance(v, list) and v and isinstance(v[0], dict) and "__ndarray__" in v[0]:
        return [_decode_tricks(x) for x in v]
    return v


def _as_kpt_array(pred, num_kpts):
    """Return keypoints as (K,3) float array [x, y, conf] from the many formats in the wild."""
    kp = _decode_tricks(pred.get("keypoints"))
    if kp is None:
        raise ValueError("prediction has no 'keypoints'")
    arr = np.asarray(kp, dtype=float)
    while arr.ndim > 2 and arr.shape[0] == 1:   # MMPose pred_instances: (1, K, 2)
        arr = arr[0]
    scores = _decode_tricks(pred.get("keypoint_scores"))
    if scores is not None:
        scores = np.asarray(scores, dtype=float).reshape(-1)
    if arr.ndim == 1:
        if arr.size == 3 * num_kpts:
            arr = arr.reshape(num_kpts, 3)
        elif arr.size == 2 * num_kpts:
            arr = arr.reshape(num_kpts, 2)
        else:
            raise ValueError(f"keypoints of length {arr.size}; expected {3 * num_kpts} or {2 * num_kpts}")
    if arr.ndim != 2 or arr.shape[0] != num_kpts or arr.shape[1] not in (2, 3):
        raise ValueError(f"keypoints of shape {arr.shape}; expected ({num_kpts}, 2|3)")
    if arr.shape[1] == 2:
        if scores is None:
            conf = np.ones(num_kpts)
            pred.setdefault("_warn", []).append("no keypoint confidences; using 1.0")
        else:
            if scores.size != num_kpts:
                raise ValueError(f"keypoint_scores of length {scores.size}")
            conf = scores
        arr = np.concatenate([arr, conf[:, None]], axis=1)
    if not np.isfinite(arr[:, :2]).all():
        raise ValueError("non-finite keypoint coordinates")
    arr[:, 2] = np.nan_to_num(arr[:, 2], nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _nms_area(pred, kpts):
    """Area used only by OKS-NMS (matches upstream when the file carries 'scale')."""
    if "area" in pred and pred["area"] is not None:
        return float(np.asarray(pred["area"], dtype=float).reshape(-1)[0])
    if "scale" in pred and pred["scale"] is not None and np.asarray(pred["scale"]).size >= 2:
        s = np.asarray(pred["scale"], dtype=float).reshape(-1)[:2]
        return float(np.prod(s * 200.0))            # upstream: np.prod(s*200, 1)
    for key in ("bbox_scale", "bbox_scales"):
        if key in pred and pred[key] is not None and np.asarray(pred[key]).size >= 2:
            return float(np.prod(np.asarray(pred[key], dtype=float).reshape(-1)[:2]))
    if "bbox" in pred and pred["bbox"] is not None:
        b = np.asarray(pred["bbox"], dtype=float).reshape(-1)
        if b.size >= 4:
            return float(b[2] * b[3])
    x, y = kpts[:, 0], kpts[:, 1]
    return float(max((x.max() - x.min()) * (y.max() - y.min()), 1.0))


def load_predictions(pred, gt):
    """Normalise a prediction file into a list of dicts:
    {image_id, category_id, kpts (K,3), score_orig, box_score_field, nms_area}.
    Accepts a list of COCO results, or a dict wrapping one ('annotations'/'results'/'predictions')."""
    data, path = _load_json(pred)
    if isinstance(data, dict):
        for key in ("annotations", "results", "predictions", "preds"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            raise ValueError("prediction JSON is a dict without a list of results")
    if not isinstance(data, list):
        raise ValueError("prediction JSON must be a list of results")
    out, unresolved, warn = [], [], []
    for i, p in enumerate(data):
        if not isinstance(p, dict):
            raise ValueError(f"prediction #{i} is not an object")
        ref = p.get("image_id")
        if ref is None:
            ref = p.get("file_name", p.get("img_path", p.get("image")))
        img_id = gt.resolve_image_id(ref)
        if img_id is None:
            unresolved.append(ref)
            continue
        p = dict(p)
        kpts = _as_kpt_array(p, gt.num_kpts)
        warn.extend(p.pop("_warn", []))
        score = _decode_tricks(p.get("score"))
        score = None if score is None else float(np.asarray(score, dtype=float).reshape(-1)[0])
        box_score = p.get("bbox_score", p.get("box_score"))
        box_score = None if box_score is None else float(np.asarray(box_score, dtype=float).reshape(-1)[0])
        cat = p.get("category_id", gt.cat_id)
        if int(cat) != gt.cat_id:
            warn.append(f"category_id {cat} mapped to GT person category {gt.cat_id}")
        out.append({
            "image_id": img_id, "category_id": gt.cat_id, "kpts": kpts,
            "score_orig": score, "box_score_field": box_score, "nms_area": _nms_area(p, kpts),
            "src_index": i,
        })
    info = {
        "path": path, "sha256": sha256_file(path) if path else None,
        "n_in": len(data), "n_resolved": len(out), "n_dropped_not_in_gt": len(unresolved),
        "dropped_examples": [str(u) for u in unresolved[:5]],
    }
    return out, info, sorted(set(warn))


def rescore_value(conf, box_score=1.0, thr=DEFAULT_KPT_THR):
    """Upstream rule: mean of confidences strictly above thr (0 if none), times box score."""
    conf = np.asarray(conf, dtype=float)
    valid = conf > thr
    kpt_score = float(conf[valid].mean()) if valid.any() else 0.0
    return kpt_score * float(box_score)


def _oks_iou_upstream(g, d, a_g, a_d, sigmas=COCO_SIGMAS):
    """Exact copy of upstream lib/nms/nms.py::oks_iou with in_vis_thre=None."""
    vars_ = (sigmas * 2) ** 2
    xg, yg = g[0::3], g[1::3]
    ious = np.zeros((d.shape[0]))
    for n_d in range(0, d.shape[0]):
        xd, yd = d[n_d, 0::3], d[n_d, 1::3]
        dx, dy = xd - xg, yd - yg
        e = (dx ** 2 + dy ** 2) / vars_ / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        ious[n_d] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0
    return ious


def oks_nms(instances, thr=DEFAULT_NMS_THR):
    """Exact copy of upstream greedy oks_nms; returns kept indices (in keep order)."""
    if len(instances) == 0:
        return []
    scores = np.array([p["score"] for p in instances])
    kpts = np.array([p["kpts"].flatten() for p in instances])
    areas = np.array([p["nms_area"] for p in instances])
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        ovr = _oks_iou_upstream(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]])
        inds = np.where(ovr <= thr)[0]
        order = order[inds + 1]
    return keep


def postprocess(preds, rescore=True, kpt_thr=DEFAULT_KPT_THR, nms_thr=DEFAULT_NMS_THR,
                nms="oks", box_score="one"):
    """Assign the evaluation score to each instance, then per-image OKS-NMS.
    Returns a list of instance dicts with 'score' (used for ranking) and 'score_rescored'."""
    by_img = OrderedDict()
    for p in preds:
        q = dict(p)
        bs = 1.0 if box_score == "one" else (1.0 if p["box_score_field"] is None else p["box_score_field"])
        q["score_rescored"] = rescore_value(p["kpts"][:, 2], bs, kpt_thr)
        if rescore:
            q["score"] = q["score_rescored"]
        else:
            if p["score_orig"] is None:
                raise ValueError("--no-rescore needs a 'score' on every prediction")
            q["score"] = p["score_orig"]
        by_img.setdefault(q["image_id"], []).append(q)
    out = []
    for img_id, inst in by_img.items():
        if nms == "oks" and len(inst) > 1:
            keep = oks_nms(inst, nms_thr)
            inst = [inst[k] for k in keep]
        out.extend(inst)
    return out


def _to_coco_results(instances):
    """Clean COCO result dicts. Only image_id/category_id/keypoints/score are passed, so that
    loadRes always takes its keypoint branch (det area = keypoint extent) for every framework."""
    return [{"image_id": int(p["image_id"]), "category_id": int(p["category_id"]),
             "keypoints": [float(v) for v in p["kpts"].reshape(-1)], "score": float(p["score"])}
            for p in instances]


# ----------------------------------------------------------------------------- COCOeval
STRICT_THRS = (0.80, 0.85, 0.90, 0.95)   # "AP_80_95": where a ~98.5-AP50 baseline has its headroom


def per_threshold(E):
    """AP and AR at each OKS threshold (area 'all', maxDets 20) from COCOeval's own arrays,
    averaged exactly like COCOeval.summarize (entries == -1 excluded)."""
    thrs = [round(float(t), 2) for t in E.params.iouThrs]
    a = E.params.areaRngLbl.index("all")
    m = len(E.params.maxDets) - 1
    prec = E.eval["precision"][:, :, :, a, m]      # (T, R, K)
    rec = E.eval["recall"][:, :, a, m]             # (T, K)
    ap, ar = OrderedDict(), OrderedDict()
    for t, thr in enumerate(thrs):
        p, r = prec[t], rec[t]
        ap[f"{thr:.2f}"] = float(np.mean(p[p > -1])) if (p > -1).any() else None
        ar[f"{thr:.2f}"] = float(np.mean(r[r > -1])) if (r > -1).any() else None
    sel = [t for t, thr in enumerate(thrs) if any(abs(thr - s) < 1e-9 for s in STRICT_THRS)]
    p = prec[sel]
    strict = float(np.mean(p[p > -1])) if (p > -1).any() else None
    return ap, ar, strict


def run_cocoeval(coco_gt, results, verbose=False):
    """Run COCOeval on a list of clean COCO result dicts.
    Returns (COCOeval | None, metrics, ap_per_thr, ar_per_thr)."""
    if len(results) == 0:
        m = OrderedDict((k, 0.0) for k in METRIC_NAMES)
        m["AP_80_95"] = 0.0
        return None, m, OrderedDict(), OrderedDict()
    with _quiet(not verbose):
        coco_dt = coco_gt.loadRes(copy.deepcopy(results))
        E = COCOeval(coco_gt, coco_dt, "keypoints")
        E.params.useSegm = None   # as upstream (deprecated no-op)
        E.evaluate()
        E.accumulate()
        E.summarize()
    m = OrderedDict((k, _none_if_neg(v)) for k, v in zip(METRIC_NAMES, E.stats))
    ap_t, ar_t, strict = per_threshold(E)
    m["AP_80_95"] = strict
    return E, m, ap_t, ar_t


def per_image_records(gt, E, instances):
    """One record per GT annotation: best-OKS detection, its scores and per-keypoint errors."""
    sig = COCO_SIGMAS
    vars_ = (sig * 2) ** 2
    dets_by_img = defaultdict(list)
    for p in instances:
        dets_by_img[p["image_id"]].append(p)
    recs = []
    for img_id in gt.img_ids:
        im = gt.images[img_id]
        dets = dets_by_img.get(img_id, [])
        for ann in gt.anns_by_img.get(img_id, []):
            g = np.asarray(ann["keypoints"], dtype=float).reshape(-1, 3)
            vis = g[:, 2]
            area = float(ann["area"])
            norm = gt.bbox_norm(ann)
            rec = OrderedDict(
                image_id=img_id, file_name=im.get("file_name"), is_synthetic=im.get("is_synthetic"),
                width=im.get("width"), height=im.get("height"), ann_id=ann["id"],
                bbox=[float(v) for v in ann["bbox"]], area=area, norm=norm,
                gt_kpts=g.tolist(), n_det=len(dets))
            best, best_oks = None, -1.0
            for p in dets:
                oks = oks_single(g, p["kpts"], area, ann["bbox"])
                if oks > best_oks:
                    best, best_oks = p, oks
            if best is None:
                rec.update(oks=0.0, score=None, score_orig=None, pred_kpts=None,
                           kpt_err_norm=[None] * len(vis), kpt_oks=[None] * len(vis))
            else:
                d = best["kpts"]
                err = np.hypot(d[:, 0] - g[:, 0], d[:, 1] - g[:, 1])
                e = err ** 2 / vars_ / (area + np.spacing(1)) / 2
                lab = vis > 0
                rec.update(
                    oks=float(best_oks), score=float(best["score"]),
                    score_rescored=float(best["score_rescored"]),
                    score_orig=None if best["score_orig"] is None else float(best["score_orig"]),
                    pred_kpts=d.tolist(),
                    kpt_err_norm=[float(x / norm) if l else None for x, l in zip(err, lab)],
                    kpt_oks=[float(np.exp(-x)) if l else None for x, l in zip(e, lab)])
            recs.append(rec)
    return recs


def oks_single(g, d, area, bbox):
    """OKS between one GT (K,3 with visibility) and one detection (K,3), exactly as
    COCOeval.computeOks (including its no-visible-keypoint branch)."""
    vars_ = (COCO_SIGMAS * 2) ** 2
    xg, yg, vg = g[:, 0], g[:, 1], g[:, 2]
    xd, yd = d[:, 0], d[:, 1]
    k1 = np.count_nonzero(vg > 0)
    if k1 > 0:
        dx, dy = xd - xg, yd - yg
    else:
        x0, x1 = bbox[0] - bbox[2], bbox[0] + bbox[2] * 2
        y0, y1 = bbox[1] - bbox[3], bbox[1] + bbox[3] * 2
        z = np.zeros(len(vg))
        dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
        dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)
    e = (dx ** 2 + dy ** 2) / vars_ / (area + np.spacing(1)) / 2
    if k1 > 0:
        e = e[vg > 0]
    return float(np.sum(np.exp(-e)) / e.shape[0])


def per_keypoint_stats(recs, kpt_names):
    K = len(kpt_names)
    errs = [[] for _ in range(K)]
    vis = [[] for _ in range(K)]
    for r in recs:
        for k in range(K):
            v = r["gt_kpts"][k][2]
            if v <= 0:
                continue
            e = r["kpt_err_norm"][k]
            # an unmatched GT (no detection) counts as an infinite error -> fails every PCK
            errs[k].append(np.inf if e is None else e)
            vis[k].append(v)
    out = OrderedDict()
    all_e = []
    for k, name in enumerate(kpt_names):
        e = np.asarray(errs[k], dtype=float)
        all_e.append(e)
        fin = e[np.isfinite(e)]
        out[name] = OrderedDict(
            n=int(e.size), n_occluded_v1=int(np.sum(np.asarray(vis[k]) == 1)),
            mean_err_norm=float(fin.mean()) if fin.size else None,
            median_err_norm=float(np.median(fin)) if fin.size else None,
            **{f"pck@{t}": (float(np.mean(e < t)) if e.size else None) for t in PCK_THRESHOLDS})
    e = np.concatenate(all_e) if all_e else np.zeros(0)
    fin = e[np.isfinite(e)]
    summary = OrderedDict(
        n=int(e.size), mean_err_norm=float(fin.mean()) if fin.size else None,
        median_err_norm=float(np.median(fin)) if fin.size else None,
        **{f"pck@{t}": (float(np.mean(e < t)) if e.size else None) for t in PCK_THRESHOLDS})
    return out, summary


def oks_summary(recs):
    o = np.array([r["oks"] for r in recs], dtype=float)
    if o.size == 0:
        return {}
    return OrderedDict(n=int(o.size), mean=float(o.mean()), median=float(np.median(o)),
                       min=float(o.min()), p10=float(np.percentile(o, 10)),
                       **{f"frac_ge_{t}": float(np.mean(o >= t)) for t in (0.5, 0.75, 0.9, 0.95)})


# ----------------------------------------------------------------------------- top level
def evaluate(gt, pred, rescore=True, kpt_thr=DEFAULT_KPT_THR, nms_thr=DEFAULT_NMS_THR, nms="oks",
             box_score="one", area_rule="gt", allow_constant_area=False, primary_protocol="paper",
             with_per_image=True, verbose=False, run_id=None, split=None):
    """Evaluate one prediction file against one GT. Returns (summary dict, per-image records)."""
    G = gt if isinstance(gt, GroundTruth) else GroundTruth(gt, area_rule, allow_constant_area)
    preds, pinfo, pwarn = load_predictions(pred, G)
    warn = list(G.warnings) + pwarn
    if pinfo["n_dropped_not_in_gt"]:
        warn.append(f"dropped {pinfo['n_dropped_not_in_gt']} predictions whose image is not in the GT "
                    f"(e.g. {pinfo['dropped_examples']})")

    has_orig = len(preds) > 0 and all(p["score_orig"] is not None for p in preds)
    primary_scoring = "rescored" if rescore else "original"
    scorings = [primary_scoring]
    alt_scoring = "original" if rescore else "rescored"
    if alt_scoring == "rescored" or has_orig:
        scorings.append(alt_scoring)
    inst = {sc: postprocess(preds, sc == "rescored", kpt_thr, nms_thr, nms, box_score) for sc in scorings}
    inst_main = inst[primary_scoring]
    results, E_primary = OrderedDict(), None
    for sc in scorings:
        for proto in PROTOCOLS:
            is_primary = (sc == primary_scoring and proto == primary_protocol)
            E, m, ap_t, ar_t = run_cocoeval(G.coco(proto), _to_coco_results(inst[sc]),
                                            verbose=verbose and is_primary)
            results[(sc, proto)] = (m, ap_t, ar_t)
            if is_primary:
                E_primary = E

    imgs_with_pred = {p["image_id"] for p in inst_main}
    n_missing = sum(1 for i in G.img_ids if G.anns_by_img.get(i) and i not in imgs_with_pred)
    if n_missing:
        warn.append(f"{n_missing} GT images have no prediction (counted as misses)")
    n_scores_ties = len(inst_main) - len({round(p["score"], 12) for p in inst_main})
    if has_orig:
        d = np.array([abs(p["score_rescored"] - p["score_orig"]) for p in inst_main])
        score_rule_maxdiff = float(d.max()) if d.size else 0.0
    else:
        score_rule_maxdiff = None

    recs = per_image_records(G, E_primary, inst_main)
    per_kpt, per_kpt_sum = per_keypoint_stats(recs, G.kpt_names)
    scoring = "rescored" if rescore else "original"
    out = OrderedDict()
    out["evaluator"] = f"eval/eval_coco.py v{EVALUATOR_VERSION}"
    out["run_id"] = run_id or infer_run_id(pinfo["path"])
    out["split"] = split or infer_split(G.path)
    out["gt"] = OrderedDict(path=G.path, sha256=G.sha256, n_images=len(G.img_ids),
                            n_annotations=len(G.dataset["annotations"]), has_ann_id_0=G.has_ann_id_0,
                            area_check=G.area_check)
    out["pred"] = OrderedDict(path=pinfo["path"], sha256=pinfo["sha256"], n_in=pinfo["n_in"],
                              n_dropped_not_in_gt=pinfo["n_dropped_not_in_gt"],
                              n_after_nms=len(inst_main), n_gt_images_without_pred=n_missing,
                              has_scores=has_orig, n_score_ties=n_scores_ties,
                              max_abs_diff_file_score_vs_rule=score_rule_maxdiff)
    out["settings"] = OrderedDict(
        primary_protocol=primary_protocol, primary_scoring=scoring,
        rescore_rule=f"mean(kpt_conf > {kpt_thr}) * box_score", box_score=box_score,
        nms=nms, oks_nms_thr=nms_thr, area_rule=G.area_rule, sigmas="COCO-17 default",
        max_dets=20, iou_type="keypoints", kpt_err_normaliser="sqrt(GT bbox w*h)",
        **_versions())
    other = [p for p in PROTOCOLS if p != primary_protocol][0]
    out["metrics"] = results[(scoring, primary_protocol)][0]
    out[f"metrics_{other}"] = results[(scoring, other)][0]
    for sc in scorings[1:]:
        out[f"metrics_{sc}_scores"] = results[(sc, primary_protocol)][0]
        out[f"metrics_{sc}_scores_{other}"] = results[(sc, other)][0]
    out["ap_per_oks_thr"] = OrderedDict((pr, results[(scoring, pr)][1]) for pr in PROTOCOLS)
    out["ar_per_oks_thr"] = OrderedDict((pr, results[(scoring, pr)][2]) for pr in PROTOCOLS)
    out["id0_artifact"] = id0_diagnostics(G, inst_main, results, scoring)
    out["oks"] = oks_summary(recs)
    out["per_keypoint_summary"] = per_kpt_sum
    out["per_keypoint"] = per_kpt
    out["warnings"] = warn
    return out, (recs if with_per_image else None)


def id0_diagnostics(G, instances, results, scoring):
    """D13: where the annotation-id-0 image's detection ranks by score, and the paper-minus-fixed gap.
    Rank 1 = highest score. COCOeval breaks score ties by image order, and image 0 comes first, so
    rank = 1 + #(scores strictly greater). None when the GT has no annotation id 0."""
    if not G.has_ann_id_0:
        return None
    zero = [a for a in G.dataset["annotations"] if a["id"] == 0][0]
    img = zero["image_id"]
    scores = np.array([p["score"] for p in instances], dtype=float)
    mine = [p for p in instances if p["image_id"] == img]
    out = OrderedDict(image_id=img, file_name=G.images[img].get("file_name"), n_dets=int(scores.size))
    if mine:
        s0 = max(p["score"] for p in mine)
        best = max(mine, key=lambda p: p["score"])
        rank = 1 + int(np.sum(scores > s0))
        out.update(det_score=float(s0), score_rank=rank, rank_percentile=float(rank / max(1, scores.size)),
                   oks=oks_single(np.asarray(zero["keypoints"], float).reshape(-1, 3), best["kpts"], float(zero["area"]),
                                  zero["bbox"]))
    else:
        out.update(det_score=None, score_rank=None, rank_percentile=None, oks=None)
    mp, mf = results[(scoring, "paper")][0], results[(scoring, "fixed")][0]
    out["paper_minus_fixed"] = OrderedDict((k, (None if mp.get(k) is None or mf.get(k) is None else mp[k] - mf[k]))
                                           for k in ("AP", "AP50", "AP75", "AP_80_95", "AR"))
    return out


def _fmt(m):
    def f(v):
        return "  n/a" if v is None else f"{100 * v:5.1f}"
    return " ".join(f"{k}={f(m[k])}" for k in ("AP", "AP50", "AP75", "AR", "AR50", "AR75"))


def _write_per_image(path, recs, meta):
    if path.endswith(".csv"):
        import csv
        names = COCO_KPT_NAMES
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "file_name", "is_synthetic", "ann_id", "norm", "n_det", "oks", "score",
                        "score_orig"] + [f"err_{n}" for n in names] + [f"v_{n}" for n in names])
            for r in recs:
                w.writerow([r["image_id"], r["file_name"], r["is_synthetic"], r["ann_id"], r["norm"], r["n_det"],
                            r["oks"], r["score"], r["score_orig"]] + list(r["kpt_err_norm"]) +
                           [g[2] for g in r["gt_kpts"]])
    else:
        with open(path, "w") as f:
            json.dump({"meta": meta, "records": recs}, f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="COCO keypoint GT json (framework-neutral split file)")
    ap.add_argument("--pred", required=True, help="COCO results list (upstream or MMPose *.keypoints.json)")
    ap.add_argument("--out", required=True, help="metrics json to write")
    ap.add_argument("--per-image", default=None, help="also write per-image records (.json or .csv)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--rescore", dest="rescore", action="store_true", default=True,
                   help="(default) recompute scores = mean(kpt conf > thr) * box score")
    g.add_argument("--no-rescore", dest="rescore", action="store_false",
                   help="primary metrics use the file's own 'score' field")
    ap.add_argument("--kpt-thr", type=float, default=DEFAULT_KPT_THR)
    ap.add_argument("--nms", choices=["oks", "none"], default="oks")
    ap.add_argument("--nms-thr", type=float, default=DEFAULT_NMS_THR)
    ap.add_argument("--box-score", choices=["one", "field"], default="one",
                    help="'one' (GT boxes, default) or use a bbox_score/box_score field if present")
    ap.add_argument("--area", choices=["gt", "bbox"], default="gt",
                    help="OKS scale: GT 'area' field (default) or bbox w*h")
    ap.add_argument("--allow-constant-area", action="store_true",
                    help="do not refuse a GT whose areas are all identical (not recommended)")
    ap.add_argument("--protocol", choices=list(PROTOCOLS), default="paper",
                    help="primary matching protocol (both are always reported)")
    ap.add_argument("--run-id", default=None, help="provenance tag (default: inferred from --pred path)")
    ap.add_argument("--split", default=None, help="split tag (default: inferred from --gt filename)")
    ap.add_argument("--verbose", action="store_true", help="show pycocotools' summary table")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    out, recs = evaluate(a.gt, a.pred, rescore=a.rescore, kpt_thr=a.kpt_thr, nms_thr=a.nms_thr, nms=a.nms,
                         box_score=a.box_score, area_rule=a.area, allow_constant_area=a.allow_constant_area,
                         primary_protocol=a.protocol, with_per_image=a.per_image is not None,
                         verbose=a.verbose, run_id=a.run_id, split=a.split)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    if a.per_image:
        _write_per_image(a.per_image, recs, {k: out[k] for k in ("evaluator", "run_id", "split", "settings")}
                         | {"gt_path": out["gt"]["path"], "pred_path": out["pred"]["path"]})
    if not a.quiet:
        other = [p for p in PROTOCOLS if p != a.protocol][0]
        print(f"[eval_coco] run={out['run_id']} split={out['split']} n_img={out['gt']['n_images']} "
              f"n_det={out['pred']['n_after_nms']} scoring={out['settings']['primary_scoring']}")
        print(f"  {a.protocol:6s}: {_fmt(out['metrics'])}")
        print(f"  {other:6s}: {_fmt(out['metrics_' + other])}")
        for k in out:
            if k.startswith("metrics_") and "_scores" in k:
                print(f"  {k}: {_fmt(out[k])}")
        apt = out["ap_per_oks_thr"][a.protocol]
        if apt:
            print("  AP@OKS " + " ".join(f"{t}:{100 * v:.1f}" for t, v in apt.items() if v is not None)
                  + f"  | AP_80_95={100 * out['metrics']['AP_80_95']:.1f}")
        z = out.get("id0_artifact")
        if z and z.get("score_rank") is not None:
            print(f"  id-0 image {z['file_name']}: score rank {z['score_rank']}/{z['n_dets']} "
                  f"(percentile {100 * z['rank_percentile']:.0f}), OKS {z['oks']:.3f}; paper-fixed AP "
                  f"{100 * z['paper_minus_fixed']['AP']:+.2f}")
        s = out["per_keypoint_summary"]
        if s.get("mean_err_norm") is not None:
            print(f"  mean OKS={out['oks']['mean']:.4f}  kpt err/sqrt(bbox)={s['mean_err_norm']:.4f}  "
                  f"PCK@0.05={s['pck@0.05']:.3f} @0.1={s['pck@0.1']:.3f} @0.2={s['pck@0.2']:.3f}")
        for w in out["warnings"]:
            print(f"  WARNING: {w}")
        print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
