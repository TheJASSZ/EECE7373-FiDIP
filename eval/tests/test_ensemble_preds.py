"""Tests for eval/ensemble_preds.py."""
import copy
import json

import numpy as np
import pytest

from conftest import make_gt, make_preds
import ensemble_preds as en
import eval_coco as ev


@pytest.mark.parametrize("weighting", ["mean", "conf"])
def test_ensemble_of_identical_files_equals_input(tmp_path, weighting):
    gt = make_gt(20, seed=0)
    preds = make_preds(gt, 0.03, seed=1)
    f = tmp_path / "a.json"
    f.write_text(json.dumps(preds))
    G = ev.GroundTruth(gt)
    out, meta = en.ensemble(G, [str(f)] * 3, weighting)
    assert meta["match"] == "single" and len(out) == len(preds)
    src = {p["image_id"]: p for p in preds}
    for o in out:
        assert o["keypoints"] == src[o["image_id"]]["keypoints"]       # bit-identical
    a, _ = ev.evaluate(copy.deepcopy(gt), preds)
    b, _ = ev.evaluate(copy.deepcopy(gt), out)
    assert a["metrics"] == b["metrics"]


def test_mean_and_conf_weighting_values():
    k1 = np.array([[0.0, 0.0, 1.0]] * 17)
    k2 = np.array([[3.0, 6.0, 0.5]] * 17)
    m = en.average_instances([k1, k2], "mean")
    assert m[0].tolist() == pytest.approx([1.5, 3.0, 0.75])
    c = en.average_instances([k1, k2], "conf")
    assert c[0].tolist() == pytest.approx([1.0, 2.0, 0.75])     # weights 1 and 0.5


def test_ensemble_reduces_independent_noise(tmp_path):
    gt = make_gt(40, seed=2)
    files = []
    for s in range(4):
        f = tmp_path / f"m{s}.json"
        f.write_text(json.dumps(make_preds(gt, 0.03, seed=10 + s)))
        files.append(str(f))
    G = ev.GroundTruth(gt)
    out, _ = en.ensemble(G, files)
    single = np.mean([ev.evaluate(copy.deepcopy(gt), json.load(open(f)))[0]["oks"]["mean"] for f in files])
    ens = ev.evaluate(copy.deepcopy(gt), out)[0]["oks"]["mean"]
    assert ens > single


def test_multi_person_gtbox_and_centroid_matching(tmp_path):
    gt = make_gt(12, seed=3, multi_person_every=2)
    p1 = make_preds(gt, 0.01, seed=1)
    p2 = list(reversed(make_preds(gt, 0.01, seed=2)))            # different instance order
    f1, f2 = tmp_path / "a.json", tmp_path / "b.json"
    f1.write_text(json.dumps(p1))
    f2.write_text(json.dumps(p2))
    G = ev.GroundTruth(gt)
    for match in ("gtbox", "centroid"):
        out, meta = en.ensemble(G, [str(f1), str(f2)], match=match)
        assert meta["match"] == match and len(out) == len(gt["annotations"])
        assert all(o["n_models"] == 2 for o in out)
        m = ev.evaluate(copy.deepcopy(gt), out)[0]
        assert m["metrics_fixed"]["AP"] > 0.9          # correct pairing keeps the persons apart


def test_missing_image_in_one_file(tmp_path):
    gt = make_gt(10, seed=4)
    f1, f2 = tmp_path / "a.json", tmp_path / "b.json"
    f1.write_text(json.dumps(make_preds(gt, 0.02, seed=1)))
    f2.write_text(json.dumps(make_preds(gt, 0.02, seed=2, drop={0, 1})))
    G = ev.GroundTruth(gt)
    out, meta = en.ensemble(G, [str(f1), str(f2)])
    assert len(out) == 10 and meta["n_instances_missing_some_model"] == 2


def test_cli(tmp_path):
    gt = make_gt(8, seed=5)
    g, a = tmp_path / "gt.json", tmp_path / "a.json"
    g.write_text(json.dumps(gt))
    a.write_text(json.dumps(make_preds(gt, 0.02)))
    o = tmp_path / "ens" / "preds_test100.json"
    assert en.main(["--gt", str(g), "--preds", str(a), str(a), "--out", str(o)]) == 0
    assert len(json.loads(o.read_text())) == 8
    meta = json.loads((tmp_path / "ens" / "preds_test100.meta.json").read_text())
    assert len(meta["inputs"]) == 2 and meta["inputs"][0]["sha256"]
