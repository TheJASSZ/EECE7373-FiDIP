"""Automatic failure taxonomy for single-person keypoint predictions (pure functions, unit-tested).

All geometry uses image coordinates (x right, y down) and COCO-17 indices.

- Left-right swaps: for each L/R pair, a swap is flagged when re-assigning the two predicted joints
  to the opposite GT joints (a) more than halves their summed error, and (b) raises the instance OKS
  by > SWAP_MIN_GAIN. The pair must be resolvable: the two GT joints are > SWAP_MIN_SEP·sqrt(area)
  apart. `oks_full_swap` is the OKS with every L/R label mirrored (whole-body front/back confusion).
- Occlusion: number of GT joints with v=1 (labelled but not visible).
- Orientation (from GT): torso axis u = mid-shoulder - mid-hip.
  - In-plane angle 0 deg = head up, +/-90 = sideways, 180 = upside down.
  - Bins: upright |a|<45, sideways 45-135, inverted >135.
  - View from the sign of r = cross(u, l_sho - r_sho)/(|u||s|): front (r > 0.3, ~supine for an
    overhead camera), back (r < -0.3, ~prone), profile otherwise (~side-lying). This sign is
    rotation invariant.
- Scale: sqrt(GT bbox w*h) in pixels, and relative to sqrt(image area).
"""
import math

import numpy as np

from eval_coco import COCO_KPT_NAMES, FLIP_PAIRS, oks_single

PAIR_NAMES = {(1, 2): "eyes", (3, 4): "ears", (5, 6): "shoulders", (7, 8): "elbows", (9, 10): "wrists",
              (11, 12): "hips", (13, 14): "knees", (15, 16): "ankles"}
SWAP_RATIO = 0.5
SWAP_MIN_GAIN = 0.02
SWAP_MIN_SEP = 0.05
VIEW_THR = 0.3
L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12


def _mid(g, i, j):
    pts = [g[k, :2] for k in (i, j) if g[k, 2] > 0]
    return np.mean(pts, axis=0) if pts else None


def orientation(gt_kpts):
    """In-plane torso angle (deg), its bin, and the front/back/profile view proxy."""
    g = np.asarray(gt_kpts, dtype=float)
    sh, hip = _mid(g, L_SHO, R_SHO), _mid(g, L_HIP, R_HIP)
    out = {"angle_deg": None, "inplane": "unknown", "view": "unknown", "view_r": None}
    if sh is None or hip is None:
        return out
    u = sh - hip
    if np.hypot(*u) < 1e-6:
        return out
    ang = math.degrees(math.atan2(u[0], -u[1]))
    a = abs(ang)
    out["angle_deg"] = ang
    out["inplane"] = "upright" if a < 45 else ("sideways" if a <= 135 else "inverted")
    s = None
    if g[L_SHO, 2] > 0 and g[R_SHO, 2] > 0:
        s = g[L_SHO, :2] - g[R_SHO, :2]
    elif g[L_HIP, 2] > 0 and g[R_HIP, 2] > 0:
        s = g[L_HIP, :2] - g[R_HIP, :2]
    if s is not None and np.hypot(*s) > 1e-6:
        r = float((u[0] * s[1] - u[1] * s[0]) / (np.hypot(*u) * np.hypot(*s)))
        out["view_r"] = r
        out["view"] = "front" if r > VIEW_THR else ("back" if r < -VIEW_THR else "profile")
    return out


def _swap(pred, pairs):
    p = np.array(pred, dtype=float, copy=True)
    for i, j in pairs:
        p[[i, j]] = p[[j, i]]
    return p


def lr_swaps(gt_kpts, pred_kpts, area, bbox):
    """Per-pair swap flags + OKS gains, and the OKS with all L/R labels mirrored."""
    g = np.asarray(gt_kpts, dtype=float)
    p = np.asarray(pred_kpts, dtype=float)
    norm = math.sqrt(max(float(bbox[2]) * float(bbox[3]), 1e-12))
    oks0 = oks_single(g, p, area, bbox)
    swapped, gains = [], {}
    for (i, j) in FLIP_PAIRS:
        if g[i, 2] <= 0 or g[j, 2] <= 0:
            continue
        sep = np.hypot(*(g[i, :2] - g[j, :2])) / norm
        d_orig = (np.hypot(*(p[i, :2] - g[i, :2])) + np.hypot(*(p[j, :2] - g[j, :2]))) / norm
        d_swap = (np.hypot(*(p[i, :2] - g[j, :2])) + np.hypot(*(p[j, :2] - g[i, :2]))) / norm
        gain = oks_single(g, _swap(p, [(i, j)]), area, bbox) - oks0
        gains[PAIR_NAMES[(i, j)]] = float(gain)
        if sep > SWAP_MIN_SEP and d_swap < SWAP_RATIO * d_orig and gain > SWAP_MIN_GAIN:
            swapped.append(PAIR_NAMES[(i, j)])
    full = oks_single(g, _swap(p, FLIP_PAIRS), area, bbox)
    return {"oks": float(oks0), "oks_full_swap": float(full), "full_swap_gain": float(full - oks0),
            "swapped_pairs": swapped, "pair_gains": gains}


def scale(rec):
    w, h = float(rec["bbox"][2]), float(rec["bbox"][3])
    s = math.sqrt(max(w * h, 1e-12))
    rel = None
    if rec.get("width") and rec.get("height"):
        rel = s / math.sqrt(float(rec["width"]) * float(rec["height"]))
    return s, rel


def n_occluded(gt_kpts):
    g = np.asarray(gt_kpts, dtype=float)
    return int(np.sum(g[:, 2] == 1))


def annotate(rec):
    """Taxonomy fields for one per-image record from eval_coco.per_image_records."""
    o = orientation(rec["gt_kpts"])
    s, rel = scale(rec)
    out = {"image_id": rec["image_id"], "file_name": rec.get("file_name"), "oks": rec["oks"],
           "angle_deg": o["angle_deg"], "inplane": o["inplane"], "view": o["view"], "view_r": o["view_r"],
           "n_v1": n_occluded(rec["gt_kpts"]), "sqrt_area": s, "rel_scale": rel}
    if rec.get("pred_kpts") is not None:
        sw = lr_swaps(rec["gt_kpts"], rec["pred_kpts"], rec["area"], rec["bbox"])
        out.update(swapped_pairs=sw["swapped_pairs"], n_swapped_pairs=len(sw["swapped_pairs"]),
                   full_swap_gain=sw["full_swap_gain"], oks_full_swap=sw["oks_full_swap"])
    else:
        out.update(swapped_pairs=[], n_swapped_pairs=0, full_swap_gain=None, oks_full_swap=None)
    return out


__all__ = ["orientation", "lr_swaps", "scale", "n_occluded", "annotate", "PAIR_NAMES", "COCO_KPT_NAMES"]
