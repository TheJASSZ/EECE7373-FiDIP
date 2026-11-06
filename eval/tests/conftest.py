"""Shared mock-data helpers for the evaluator tests (no real SyRIP data needed)."""
import copy
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(EVAL_DIR)
sys.path.insert(0, EVAL_DIR)

from eval_coco import COCO_KPT_NAMES, rescore_value  # noqa: E402


def make_gt(n_images=30, seed=0, first_ann_id=1, n_v1=2, n_v0=0, constant_area=None,
            multi_person_every=0, start_image_id=0):
    """SyRIP-like COCO keypoint GT: one infant per image (optionally a second one every k images),
    area = bbox w*h, annotation id = image id + first_ann_id - start_image_id."""
    rng = np.random.default_rng(seed)
    images, anns = [], []
    ann_id = first_ann_id
    for i in range(n_images):
        img_id = start_image_id + i
        W, H = int(rng.choice([640, 960, 1280])), int(rng.choice([480, 720]))
        images.append({"id": img_id, "file_name": f"test{img_id}.jpg", "width": W, "height": H,
                       "is_synthetic": False})
        n_people = 2 if (multi_person_every and i % multi_person_every == 0) else 1
        for p in range(n_people):
            bw, bh = rng.uniform(150, 420), rng.uniform(150, 420)
            x0 = rng.uniform(0, W - bw) if n_people == 1 else (p * W / 2 + rng.uniform(0, max(W / 2 - bw, 1)))
            y0 = rng.uniform(0, H - bh)
            kp = np.zeros((17, 3))
            kp[:, 0] = x0 + rng.uniform(0.1, 0.9, 17) * bw
            kp[:, 1] = y0 + rng.uniform(0.1, 0.9, 17) * bh
            kp[:, 2] = 2
            occ = rng.choice(17, size=n_v1 + n_v0, replace=False)
            kp[occ[:n_v1], 2] = 1
            kp[occ[n_v1:], 2] = 0
            kp[kp[:, 2] == 0, :2] = 0
            area = float(constant_area) if constant_area is not None else float(bw * bh)
            anns.append({"id": ann_id, "image_id": img_id, "category_id": 1,
                         "keypoints": kp.reshape(-1).tolist(), "num_keypoints": int((kp[:, 2] > 0).sum()),
                         "bbox": [float(x0), float(y0), float(bw), float(bh)], "area": area, "iscrowd": 0})
            ann_id += 1
    cats = [{"id": 1, "name": "person", "supercategory": "person", "keypoints": list(COCO_KPT_NAMES),
             "skeleton": [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                          [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]}]
    return {"images": images, "annotations": anns, "categories": cats}


def make_preds(gt, noise=0.0, seed=1, conf=None, base_noise=None, drop=(), score="rule"):
    """Predictions = GT keypoints + noise * sqrt(area) * N(0,1) (common random numbers via base_noise).
    v=0 GT keypoints get a random location. Confidences in (0.3, 1.0) unless given."""
    rng = np.random.default_rng(seed)
    preds = []
    for j, a in enumerate(gt["annotations"]):
        if a["image_id"] in drop:
            continue
        g = np.asarray(a["keypoints"], dtype=float).reshape(17, 3)
        s = np.sqrt(a["bbox"][2] * a["bbox"][3])
        z = base_noise[j] if base_noise is not None else rng.standard_normal((17, 2))
        xy = g[:, :2] + noise * s * z
        if (g[:, 2] == 0).any():
            xy[g[:, 2] == 0] = [a["bbox"][0] + a["bbox"][2] / 2, a["bbox"][1] + a["bbox"][3] / 2]
        c = rng.uniform(0.3, 1.0, 17) if conf is None else np.asarray(conf[j], dtype=float)
        kp = np.concatenate([xy, c[:, None]], axis=1)
        d = {"image_id": a["image_id"], "category_id": 1, "keypoints": kp.reshape(-1).tolist()}
        if score == "rule":
            d["score"] = rescore_value(c)
        elif score is not None:
            d["score"] = float(score(j, c))
        preds.append(d)
    return preds


def gt_as_preds(gt, conf=None):
    out = []
    for j, a in enumerate(gt["annotations"]):
        g = np.asarray(a["keypoints"], dtype=float).reshape(17, 3).copy()
        c = np.linspace(0.5, 0.99, 17) if conf is None else conf[j]
        g[:, 2] = c
        out.append({"image_id": a["image_id"], "category_id": 1, "keypoints": g.reshape(-1).tolist(),
                    "score": rescore_value(c)})
    return out


@pytest.fixture
def gt30():
    return make_gt(30, seed=0)


@pytest.fixture
def deep():
    return copy.deepcopy
