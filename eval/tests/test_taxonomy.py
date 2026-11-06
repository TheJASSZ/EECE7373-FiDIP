"""Tests for eval/taxonomy.py on constructed poses."""
import math

import numpy as np
import pytest

import taxonomy as tx

# A frontal, upright, roughly symmetric person (image coords, y down). Subject's left = image right.
BASE = np.array([
    [100, 40], [108, 34], [92, 34], [116, 38], [84, 38],      # nose, eyes, ears
    [125, 70], [75, 70], [135, 110], [65, 110], [140, 150], [60, 150],   # shoulders, elbows, wrists
    [115, 160], [85, 160], [118, 220], [82, 220], [120, 280], [80, 280],  # hips, knees, ankles
], dtype=float)
BBOX = [50.0, 20.0, 100.0, 270.0]
AREA = BBOX[2] * BBOX[3]


def gt(v=2):
    return np.concatenate([BASE, np.full((17, 1), v, float)], axis=1)


def pred_from(xy, conf=0.9):
    return np.concatenate([xy, np.full((17, 1), conf)], axis=1)


def rotate(xy, deg, c=(100, 160)):
    t = math.radians(deg)
    R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return (xy - c) @ R.T + c


def test_orientation_upright_front():
    o = tx.orientation(gt())
    assert abs(o["angle_deg"]) < 1e-9 and o["inplane"] == "upright" and o["view"] == "front"


@pytest.mark.parametrize("deg,bin_", [(90, "sideways"), (-90, "sideways"), (180, "inverted"), (30, "upright"),
                                      (150, "inverted"), (100, "sideways")])
def test_orientation_rotation_bins_and_view_invariance(deg, bin_):
    g = gt()
    g[:, :2] = rotate(BASE, deg)
    o = tx.orientation(g)
    assert o["inplane"] == bin_
    assert o["view"] == "front"                        # view sign is rotation invariant
    assert abs(abs(o["angle_deg"]) - abs(deg)) < 1e-6


def test_orientation_back_view_is_mirror():
    g = gt()
    g[:, 0] = 200 - g[:, 0]                            # mirror: left shoulder now on image left
    assert tx.orientation(g)["view"] == "back"


def test_orientation_profile_when_shoulders_overlap():
    g = gt()
    g[[5, 6], 0] = 100.0                               # shoulders stacked on the torso axis
    g[[11, 12], 0] = 100.0
    g[6, 1] += 3
    assert tx.orientation(g)["view"] == "profile"


def test_lr_swap_detects_swapped_wrists_only():
    p = pred_from(BASE.copy())
    p[[9, 10]] = p[[10, 9]]
    r = tx.lr_swaps(gt(), p, AREA, BBOX)
    assert r["swapped_pairs"] == ["wrists"]
    assert r["pair_gains"]["wrists"] > 0.05
    assert r["oks"] < 0.9


def test_lr_swap_full_mirror_detected_and_gain():
    p = pred_from(BASE.copy())
    for i, j in tx.FLIP_PAIRS:
        p[[i, j]] = p[[j, i]]
    r = tx.lr_swaps(gt(), p, AREA, BBOX)
    assert set(r["swapped_pairs"]) == set(tx.PAIR_NAMES.values())
    assert r["oks_full_swap"] == pytest.approx(1.0) and r["full_swap_gain"] > 0.5


def test_no_swap_for_small_jitter():
    rng = np.random.default_rng(0)
    p = pred_from(BASE + rng.normal(0, 1.5, BASE.shape))
    r = tx.lr_swaps(gt(), p, AREA, BBOX)
    assert r["swapped_pairs"] == [] and r["full_swap_gain"] < 0


def test_unresolvable_pair_not_flagged():
    g = gt()
    g[3, :2] = g[4, :2] = [100, 38]                    # both ears at the same GT point
    p = pred_from(g[:, :2].copy())
    p[3, :2] += [6, 0]
    p[4, :2] -= [6, 0]
    assert "ears" not in tx.lr_swaps(g, p, AREA, BBOX)["swapped_pairs"]


def test_annotate_record():
    g = gt()
    g[[7, 8], 2] = 1
    rec = {"image_id": 3, "file_name": "test3.jpg", "oks": 0.9, "gt_kpts": g.tolist(),
           "pred_kpts": pred_from(BASE).tolist(), "area": AREA, "bbox": BBOX, "width": 400, "height": 300}
    a = tx.annotate(rec)
    assert a["n_v1"] == 2 and a["inplane"] == "upright" and a["view"] == "front"
    assert a["sqrt_area"] == pytest.approx(math.sqrt(AREA))
    assert a["rel_scale"] == pytest.approx(math.sqrt(AREA) / math.sqrt(400 * 300))
    assert a["n_swapped_pairs"] == 0
