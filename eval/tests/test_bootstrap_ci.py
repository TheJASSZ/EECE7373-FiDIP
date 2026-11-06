"""Tests for eval/bootstrap_ci.py: the fast path must reproduce COCOeval exactly."""
import copy
import json

import numpy as np
import pytest

from conftest import gt_as_preds, make_gt, make_preds
import bootstrap_ci as bc
import eval_coco as ev


def _messy(seed=0, first_ann_id=1):
    """GT with multi-person images, preds with misses, duplicates and score ties."""
    gt = make_gt(25, seed=seed, first_ann_id=first_ann_id, multi_person_every=4, n_v1=2)
    preds = make_preds(gt, 0.04, seed=seed + 1, drop={3, 7})
    rng = np.random.default_rng(seed)
    extra = []
    for p in preds[:6]:                       # near-duplicates (some survive NMS, some not)
        q = copy.deepcopy(p)
        q["keypoints"] = (np.asarray(q["keypoints"]) + rng.normal(0, 6, 51) * np.tile([1, 1, 0], 17)).tolist()
        extra.append(q)
    preds += extra
    for p in preds[10:14]:                    # exact score ties via identical confidences
        kp = np.asarray(p["keypoints"]).reshape(17, 3)
        kp[:, 2] = 0.77
        p["keypoints"] = kp.reshape(-1).tolist()
    return gt, preds


@pytest.mark.parametrize("protocol,first", [("paper", 1), ("paper", 0), ("fixed", 0)])
def test_fast_path_full_set_equals_cocoeval(protocol, first):
    gt, preds = _messy(first_ann_id=first)
    G = ev.GroundTruth(copy.deepcopy(gt))
    M = bc.PerImageEval(G, copy.deepcopy(preds), protocol)
    out, _ = ev.evaluate(copy.deepcopy(gt), copy.deepcopy(preds), primary_protocol=protocol)
    fast = M.stats()
    for k in bc.STAT_NAMES:
        assert fast[k] == out["metrics"][k], k              # bit-identical
    assert M.self_check["passed"]


def _duplicated_dataset(gt, preds, sample):
    """Build the resampled multiset explicitly as a new COCO dataset (new image/ann ids in sample order)."""
    imgs = {im["id"]: im for im in gt["images"]}
    anns = {}
    for a in gt["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    pr = {}
    for p in preds:
        pr.setdefault(p["image_id"], []).append(p)
    order = sorted(imgs)
    new_imgs, new_anns, new_preds = [], [], []
    aid = 1
    for new_id, k in enumerate(sample):
        old = order[k]
        new_imgs.append(dict(imgs[old], id=new_id, file_name=f"dup{new_id}.jpg"))
        for a in anns.get(old, []):
            new_anns.append(dict(a, id=aid, image_id=new_id))
            aid += 1
        for p in pr.get(old, []):
            new_preds.append(dict(p, image_id=new_id))
    return dict(gt, images=new_imgs, annotations=new_anns), new_preds


def test_fast_path_resample_equals_cocoeval_on_duplicated_dataset():
    gt, preds = _messy(seed=3, first_ann_id=1)
    G = ev.GroundTruth(copy.deepcopy(gt))
    M = bc.PerImageEval(G, copy.deepcopy(preds), "paper")
    rng = np.random.default_rng(5)
    for _ in range(8):
        sample = rng.integers(0, len(gt["images"]), len(gt["images"]))
        g2, p2 = _duplicated_dataset(gt, preds, sample)
        ref, _ = ev.evaluate(g2, p2)
        fast = M.stats(sample)
        for k in bc.STAT_NAMES:
            assert fast[k] == pytest.approx(ref["metrics"][k], abs=1e-12), k


def test_paired_identical_files_zero_delta(tmp_path):
    gt = make_gt(30, seed=1)
    preds = make_preds(gt, 0.03, seed=2)
    pa = tmp_path / "a.json"
    pa.write_text(json.dumps(preds))
    G = ev.GroundTruth(gt)
    r = bc.compare(G, [str(pa)], [str(pa)], B=300, seed=0)
    assert r["delta"]["AP"]["observed"] == 0.0
    assert r["delta"]["AP"]["ci95"] == [0.0, 0.0]
    assert r["p_one_sided_a_gt_b"] == 1.0
    assert r["per_image_consistency"]["ties"] == 30


def test_paired_detects_clear_improvement(tmp_path):
    gt = make_gt(60, seed=2)
    rng = np.random.default_rng(0)
    base = rng.standard_normal((60, 17, 2))
    conf = rng.uniform(0.3, 1, (60, 17))
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(json.dumps(make_preds(gt, 0.02, base_noise=base, conf=conf)))
    bad.write_text(json.dumps(make_preds(gt, 0.04, base_noise=base, conf=conf)))
    G = ev.GroundTruth(gt)
    r = bc.compare(G, [str(good)], [str(bad)], B=1000, seed=0)
    assert r["delta"]["AP"]["observed"] > 0.05
    assert r["delta"]["AP"]["ci95"][0] > 0
    assert r["p_one_sided_a_gt_b"] < 0.01
    assert r["per_image_consistency"]["wins"] == 60      # every image's OKS improves (common noise)
    assert r["a"]["self_check"][0]["passed"] and r["b"]["self_check"][0]["passed"]


def test_bootstrap_is_deterministic_given_seed(tmp_path):
    gt = make_gt(20, seed=4)
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(make_preds(gt, 0.02, seed=1)))
    b.write_text(json.dumps(make_preds(gt, 0.03, seed=2)))
    G = ev.GroundTruth(gt)
    r1 = bc.compare(G, [str(a)], [str(b)], B=200, seed=7)
    r2 = bc.compare(G, [str(a)], [str(b)], B=200, seed=7)
    assert r1["delta"] == r2["delta"] and r1["p_one_sided_a_gt_b"] == r2["p_one_sided_a_gt_b"]


def test_multiseed_identical_seeds_match_single(tmp_path):
    gt = make_gt(25, seed=5)
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(make_preds(gt, 0.02, seed=1)))
    b.write_text(json.dumps(make_preds(gt, 0.03, seed=2)))
    G = ev.GroundTruth(gt)
    single = bc.compare(G, [str(a)], [str(b)], B=200, seed=0)
    multi = bc.compare(G, [str(a)] * 3, [str(b)] * 2, B=200, seed=0)
    assert multi["mode"] == "multiseed"
    assert multi["a"]["std_across_runs"]["AP"] == 0.0
    assert multi["delta"]["AP"]["ci95"] == pytest.approx(single["delta"]["AP"]["ci95"], abs=1e-15)
    assert multi["seed_pairs_a_gt_b"] in (0.0, 1.0)


def test_multiseed_mean_and_std(tmp_path):
    gt = make_gt(25, seed=6)
    files = []
    for s in range(3):
        f = tmp_path / f"a{s}.json"
        f.write_text(json.dumps(make_preds(gt, 0.02 + 0.005 * s, seed=s)))
        files.append(str(f))
    G = ev.GroundTruth(gt)
    r = bc.compare(G, files, files[:1], B=100, seed=0, hierarchical=True)
    aps = [ev.evaluate(gt, json.load(open(f)))[0]["metrics"]["AP"] for f in files]
    assert r["a"]["mean"]["AP"] == pytest.approx(np.mean(aps), abs=1e-12)
    assert r["a"]["std_across_runs"]["AP"] == pytest.approx(np.std(aps, ddof=1), abs=1e-12)
    assert "hierarchical" in r


def test_holm_adjustment():
    p = [0.01, 0.04, 0.03, 0.005, 0.2]
    adj = bc.holm(p)
    # sorted: 0.005*5=0.025, 0.01*4=0.04, 0.03*3=0.09, 0.04*2=0.08->max 0.09, 0.2*1=0.2
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.025, 0.2])
    assert bc.holm([0.5, 0.6]) == pytest.approx([1.0, 1.0])


def test_contrasts_mode(tmp_path):
    gt = make_gt(30, seed=7)
    rng = np.random.default_rng(1)
    base = rng.standard_normal((30, 17, 2))
    conf = rng.uniform(0.3, 1, (30, 17))
    f = {}
    for name, s in [("best", 0.015), ("mid", 0.025), ("base", 0.035)]:
        f[name] = tmp_path / f"{name}.json"
        f[name].write_text(json.dumps(make_preds(gt, s, base_noise=base, conf=conf)))
    spec = {"B": 300, "contrasts": [
        {"name": "best>base", "a": str(f["best"]), "b": str(f["base"])},
        {"name": "mid>base", "a": [str(f["mid"])], "b": [str(f["base"])]},
        {"name": "base>best", "a": str(f["base"]), "b": str(f["best"])}]}
    G = ev.GroundTruth(gt)
    out = bc.run_contrasts(G, spec)
    ps = [c["p_one_sided_a_gt_b"] for c in out["contrasts"]]
    assert [c["p_holm"] for c in out["contrasts"]] == pytest.approx(bc.holm(ps))
    assert out["contrasts"][0]["reject_h0_holm"] and not out["contrasts"][2]["reject_h0_holm"]


def test_cli_paired_and_single(tmp_path):
    gt = make_gt(20, seed=8)
    g, a, b = tmp_path / "gt.json", tmp_path / "a.json", tmp_path / "b.json"
    g.write_text(json.dumps(gt))
    a.write_text(json.dumps(make_preds(gt, 0.02, seed=1)))
    b.write_text(json.dumps(gt_as_preds(gt)))
    o = tmp_path / "ci.json"
    assert bc.main(["paired", "--gt", str(g), "--a", str(a), "--b", str(b), "--out", str(o), "--B", "100"]) == 0
    r = json.loads(o.read_text())
    assert r["delta"]["AP"]["observed"] < 0 and r["settings"]["ci"] == "percentile"
    assert bc.main(["single", "--gt", str(g), "--a", str(b), "--out", str(o), "--B", "50"]) == 0
    r = json.loads(o.read_text())
    assert r["observed"]["AP"] == pytest.approx(1.0) and r["boot_ci95"]["AP"] == pytest.approx([1.0, 1.0])
