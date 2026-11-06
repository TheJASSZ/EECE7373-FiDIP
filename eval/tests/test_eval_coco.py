"""Unit tests for eval/eval_coco.py (run: python -m pytest eval/tests -q)."""
import ast
import copy
import json
import os

import numpy as np
import pytest

from conftest import REPO, gt_as_preds, make_gt, make_preds
import eval_coco as ev
from eval_coco import GroundTruth, evaluate

STD = ["AP", "AP50", "AP75", "AR", "AR50", "AR75"]


def _m(gt, preds, **kw):
    out, recs = evaluate(copy.deepcopy(gt), copy.deepcopy(preds), **kw)
    return out, recs


# ------------------------------------------------------------------ GT as prediction
def test_gt_as_prediction_gives_ap_one(gt30):
    out, recs = _m(gt30, gt_as_preds(gt30))
    for k in STD + ["APl", "ARl", "AP_80_95"]:
        assert out["metrics"][k] == pytest.approx(1.0, abs=1e-12), k
    assert out["metrics"]["APm"] is None            # no medium-size GT -> COCOeval -1 -> None
    assert out["metrics_fixed"] == out["metrics"]   # no annotation id 0 -> protocols agree
    assert all(r["oks"] == pytest.approx(1.0) for r in recs)
    assert out["per_keypoint_summary"]["mean_err_norm"] == pytest.approx(0.0, abs=1e-12)
    assert out["per_keypoint_summary"]["pck@0.05"] == 1.0
    assert all(v == pytest.approx(1.0) for v in out["ap_per_oks_thr"]["paper"].values())


def test_annotation_id_zero_artifact_paper_vs_fixed():
    gt = make_gt(10, seed=3, first_ann_id=0)          # annotation ids 0..9 (like SyRIP Test100)
    out, _ = _m(gt, gt_as_preds(gt))
    assert out["gt"]["has_ann_id_0"] is True
    assert out["metrics"]["AR"] == pytest.approx(0.9)          # the id-0 GT can never be matched
    assert out["metrics"]["AP"] < 0.99
    for k in STD:
        assert out["metrics_fixed"][k] == pytest.approx(1.0), k
    assert any("annotation id 0" in w for w in out["warnings"])
    out2, _ = _m(gt, gt_as_preds(gt), primary_protocol="fixed")
    assert out2["metrics"] == out["metrics_fixed"] and out2["metrics_paper"] == out["metrics"]


# ------------------------------------------------------------------ monotone degradation
def test_monotone_degradation_under_noise():
    gt = make_gt(40, seed=5)
    rng = np.random.default_rng(11)
    base = rng.standard_normal((len(gt["annotations"]), 17, 2))     # common random numbers
    conf = rng.uniform(0.3, 1.0, (len(gt["annotations"]), 17))
    sigmas = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16]
    res = [_m(gt, make_preds(gt, s, base_noise=base, conf=conf))[0] for s in sigmas]
    ap = [r["metrics"]["AP"] for r in res]
    oks = [r["oks"]["mean"] for r in res]
    err = [r["per_keypoint_summary"]["mean_err_norm"] for r in res]
    pck = [r["per_keypoint_summary"]["pck@0.1"] for r in res]
    assert ap[0] == pytest.approx(1.0)
    assert all(a >= b - 1e-12 for a, b in zip(ap, ap[1:])), ap
    assert ap[-1] < 0.5 < ap[1]
    assert all(a > b for a, b in zip(oks, oks[1:])), oks
    assert all(a < b for a, b in zip(err, err[1:])), err
    assert all(a >= b for a, b in zip(pck, pck[1:])), pck


# ------------------------------------------------------------------ format robustness
def _variants(preds, gt):
    names = {im["id"]: im["file_name"] for im in gt["images"]}
    v = {}
    v["str_id"] = [dict(p, image_id=str(p["image_id"])) for p in preds]
    v["file_stem"] = [dict(p, image_id=os.path.splitext(names[p["image_id"]])[0]) for p in preds]
    v["file_name"] = [dict(p, image_id=names[p["image_id"]]) for p in preds]
    v["img_path_only"] = [{k: x for k, x in dict(p, img_path=f"/data/x/{names[p['image_id']]}").items()
                           if k != "image_id"} for p in preds]
    v["nested_17x3"] = [dict(p, keypoints=np.asarray(p["keypoints"]).reshape(17, 3).tolist()) for p in preds]
    v["xy_plus_scores"] = [dict(p, keypoints=np.asarray(p["keypoints"]).reshape(17, 3)[:, :2].reshape(-1).tolist(),
                                keypoint_scores=np.asarray(p["keypoints"]).reshape(17, 3)[:, 2].tolist())
                           for p in preds]
    v["mmpose_pred_instances"] = [dict(p, keypoints=[np.asarray(p["keypoints"]).reshape(17, 3)[:, :2].tolist()],
                                       keypoint_scores=[np.asarray(p["keypoints"]).reshape(17, 3)[:, 2].tolist()])
                                  for p in preds]
    v["extra_fields"] = [dict(p, center=[1.0, 2.0], scale=[3.0, 4.0], bbox=[0, 0, 5, 5], area=25.0, id=99,
                              track_id=0, bbox_score=0.5) for p in preds]
    v["no_category"] = [{k: x for k, x in p.items() if k != "category_id"} for p in preds]
    v["float_id"] = [dict(p, image_id=float(p["image_id"])) for p in preds]
    v["dict_wrapper"] = {"annotations": preds}
    v["shuffled"] = [preds[i] for i in np.random.default_rng(0).permutation(len(preds))]
    return v


def test_format_robustness(gt30):
    preds = make_preds(gt30, noise=0.03, seed=2)
    ref, ref_recs = _m(gt30, preds)
    assert 0.3 < ref["metrics"]["AP"] < 0.99
    for name, var in _variants(preds, gt30).items():
        out, recs = _m(gt30, var)
        assert out["metrics"] == ref["metrics"], name
        assert out["metrics_fixed"] == ref["metrics_fixed"], name
        assert [r["oks"] for r in recs] == [r["oks"] for r in ref_recs], name
        assert out["pred"]["n_dropped_not_in_gt"] == 0, name


def test_predictions_for_unknown_images_are_dropped(gt30):
    preds = make_preds(gt30, noise=0.03, seed=2)
    extra = [dict(preds[0], image_id=10_000), dict(preds[1], image_id="train00001.jpg")]
    ref, _ = _m(gt30, preds)
    out, _ = _m(gt30, preds + extra)
    assert out["metrics"] == ref["metrics"]
    assert out["pred"]["n_dropped_not_in_gt"] == 2
    assert any("dropped 2" in w for w in out["warnings"])


def test_missing_prediction_counts_as_miss():
    gt = make_gt(10, seed=4)
    out, recs = _m(gt, gt_as_preds(gt)[1:])
    assert out["metrics"]["AR"] == pytest.approx(0.9)
    assert out["pred"]["n_gt_images_without_pred"] == 1
    assert recs[0]["oks"] == 0.0 and recs[0]["n_det"] == 0
    assert out["per_keypoint_summary"]["pck@0.2"] < 1.0      # unmatched GT keypoints fail PCK


def test_bad_keypoint_length_raises(gt30):
    preds = make_preds(gt30, noise=0.01)
    preds[0]["keypoints"] = preds[0]["keypoints"][:50]
    with pytest.raises(ValueError):
        _m(gt30, preds)


# ------------------------------------------------------------------ rescoring
def test_rescoring_invariance_when_file_used_same_rule(gt30):
    preds = make_preds(gt30, noise=0.03, seed=7)
    out, _ = _m(gt30, preds)
    assert out["metrics_original_scores"] == out["metrics"]
    assert out["pred"]["max_abs_diff_file_score_vs_rule"] < 1e-12


def test_rescoring_ignores_file_scores(gt30):
    rng = np.random.default_rng(0)
    preds = make_preds(gt30, noise=0.03, seed=7)
    junk = [dict(p, score=float(rng.uniform())) for p in preds]
    ref, _ = _m(gt30, preds)
    out, _ = _m(gt30, junk)
    assert out["metrics"] == ref["metrics"]                       # rescored: file score irrelevant
    assert out["metrics_original_scores"] != ref["metrics"]       # un-rescored differs (junk ranking)


def test_no_rescore_invariant_to_monotone_score_transform(gt30):
    preds = make_preds(gt30, noise=0.03, seed=8)
    a, _ = _m(gt30, preds, rescore=False)
    b, _ = _m(gt30, [dict(p, score=float(np.exp(3 * p["score"]) - 7)) for p in preds], rescore=False)
    assert a["metrics"] == b["metrics"]
    assert a["settings"]["primary_scoring"] == "original"
    assert a["metrics_rescored_scores"] == a["metrics"]          # file scores follow the rule here


def test_rescore_invariant_to_confidence_scaling(gt30):
    rng = np.random.default_rng(1)
    conf = rng.uniform(0.4, 0.9, (len(gt30["annotations"]), 17))
    a, _ = _m(gt30, make_preds(gt30, 0.03, seed=9, conf=conf))
    b, _ = _m(gt30, make_preds(gt30, 0.03, seed=9, conf=conf * 1.1))   # stays above 0.2 threshold
    assert a["metrics"] == b["metrics"]


def test_rescore_rule_matches_upstream_loop():
    rng = np.random.default_rng(0)
    for _ in range(200):
        c = rng.uniform(0, 1, 17) * rng.choice([1.0, 0.25])
        kpt_score, valid = 0, 0
        for t in c:                      # verbatim logic of upstream syrip.py::evaluate
            if t > 0.2:
                kpt_score += t
                valid += 1
        if valid != 0:
            kpt_score = kpt_score / valid
        assert ev.rescore_value(c) == pytest.approx(kpt_score * 1.0, abs=1e-15)
    assert ev.rescore_value(np.full(17, 0.2)) == 0.0     # strictly-greater threshold, none valid


# ------------------------------------------------------------------ area guard
def test_constant_area_refused_unless_bbox_rule():
    gt = make_gt(20, seed=1, constant_area=2443298.4755617115)       # like SyRIP validate500
    preds = make_preds(gt, 0.02)
    with pytest.raises(ValueError, match="identical"):
        _m(gt, preds)
    out, _ = _m(gt, preds, area_rule="bbox")
    ref, _ = _m(make_gt(20, seed=1), preds)                          # same GT with area = w*h
    assert out["metrics"] == ref["metrics"]
    assert out["gt"]["area_check"]["rule"] == "bbox"
    assert out["gt"]["area_check"]["frac_area_equals_bbox_wh"] == 1.0
    loose, _ = _m(gt, preds, allow_constant_area=True)
    assert loose["metrics"]["AP"] > ref["metrics"]["AP"]             # huge constant area = lenient OKS


# ------------------------------------------------------------------ per-image / per-keypoint
def test_per_image_oks_equals_cocoeval_ious(gt30):
    preds = make_preds(gt30, 0.04, seed=3)
    G = GroundTruth(copy.deepcopy(gt30))
    inst = ev.postprocess(ev.load_predictions(copy.deepcopy(preds), G)[0])
    E, _, _, _ = ev.run_cocoeval(G.coco("paper"), ev._to_coco_results(inst))
    recs = ev.per_image_records(G, E, inst)
    for r in recs:
        ious = E.ious[(r["image_id"], 1)]
        assert r["oks"] == pytest.approx(float(np.max(ious)), abs=1e-12)


def test_per_keypoint_error_and_pck_exact():
    gt = make_gt(12, seed=6, n_v1=0)
    preds = gt_as_preds(gt)
    lw = ev.COCO_KPT_NAMES.index("left_wrist")
    for p, a in zip(preds, gt["annotations"]):
        s = np.sqrt(a["bbox"][2] * a["bbox"][3])
        kp = np.asarray(p["keypoints"]).reshape(17, 3)
        kp[lw, 0] += 0.15 * s          # 0.15 of sqrt(bbox area)
        p["keypoints"] = kp.reshape(-1).tolist()
    out, _ = _m(gt, preds)
    k = out["per_keypoint"]["left_wrist"]
    assert k["mean_err_norm"] == pytest.approx(0.15, abs=1e-9)
    assert (k["pck@0.05"], k["pck@0.1"], k["pck@0.2"]) == (0.0, 0.0, 1.0)
    assert out["per_keypoint"]["nose"]["pck@0.05"] == 1.0
    s = out["per_keypoint_summary"]
    assert s["pck@0.1"] == pytest.approx(16 / 17)


def test_v0_keypoints_excluded_and_v1_counted():
    gt = make_gt(10, seed=2, n_v1=3, n_v0=2)
    out, recs = _m(gt, gt_as_preds(gt))
    assert out["per_keypoint_summary"]["n"] == 10 * 15
    assert sum(v["n_occluded_v1"] for v in out["per_keypoint"].values()) == 30
    assert all(sum(e is None for e in r["kpt_err_norm"]) == 2 for r in recs)


def test_per_threshold_consistency(gt30):
    out, _ = _m(gt30, make_preds(gt30, 0.04, seed=12))
    apt = out["ap_per_oks_thr"]["paper"]
    assert list(apt) == [f"{t:.2f}" for t in np.linspace(0.5, 0.95, 10)]
    assert np.mean(list(apt.values())) == pytest.approx(out["metrics"]["AP"], abs=1e-12)
    assert apt["0.50"] == pytest.approx(out["metrics"]["AP50"], abs=1e-12)
    assert apt["0.75"] == pytest.approx(out["metrics"]["AP75"], abs=1e-12)
    strict = np.mean([apt[k] for k in ("0.80", "0.85", "0.90", "0.95")])
    assert out["metrics"]["AP_80_95"] == pytest.approx(strict, abs=1e-12)
    art = out["ar_per_oks_thr"]["paper"]
    assert np.mean(list(art.values())) == pytest.approx(out["metrics"]["AR"], abs=1e-12)


# ------------------------------------------------------------------ OKS-NMS = upstream
def _load_upstream_nms():
    path = os.path.join(REPO, "third_party", "Infant-Pose-Estimation", "lib", "nms", "nms.py")
    src = open(path).read()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("oks_iou", "oks_nms")]
    ns = {"np": np}
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)   # avoids the compiled .so imports
    return ns["oks_nms"]


def test_oks_nms_matches_upstream_exactly():
    up_nms = _load_upstream_nms()
    rng = np.random.default_rng(0)
    for trial in range(60):
        n = int(rng.integers(1, 8))
        base = rng.uniform(0, 300, (17, 2))
        inst = []
        for i in range(n):
            jitter = rng.choice([0.5, 3.0, 40.0])
            k = np.concatenate([base + rng.normal(0, jitter, (17, 2)), rng.uniform(0.2, 1, (17, 1))], 1)
            inst.append({"kpts": k, "score": float(rng.uniform()), "nms_area": float(rng.uniform(5e3, 5e4))})
        up = up_nms([{"keypoints": p["kpts"], "score": p["score"], "area": p["nms_area"]} for p in inst], 0.9)
        assert [int(i) for i in up] == ev.oks_nms(inst, 0.9), trial


def test_nms_removes_near_duplicates(gt30):
    preds = make_preds(gt30, 0.02, seed=4)
    dup = [dict(p, keypoints=(np.asarray(p["keypoints"]) + 0.01).tolist()) for p in preds[:5]]
    ref, _ = _m(gt30, preds)
    out, _ = _m(gt30, preds + dup)
    assert out["pred"]["n_after_nms"] == len(preds)
    assert out["metrics"]["AP"] == pytest.approx(ref["metrics"]["AP"], abs=2e-3)
    raw, _ = _m(gt30, preds + dup, nms="none")
    assert raw["pred"]["n_after_nms"] == len(preds) + 5


# ------------------------------------------------------------------ CLI
def test_cli_end_to_end(tmp_path, gt30):
    gp, pp = tmp_path / "test100.json", tmp_path / "runs" / "R001_fidip-w48_released" / "preds_test100.json"
    pp.parent.mkdir(parents=True)
    gp.write_text(json.dumps(gt30))
    pp.write_text(json.dumps(make_preds(gt30, 0.03)))
    out_p, per_p = tmp_path / "m.json", tmp_path / "per_image.json"
    assert ev.main(["--gt", str(gp), "--pred", str(pp), "--out", str(out_p), "--per-image", str(per_p),
                    "--quiet"]) == 0
    out = json.loads(out_p.read_text())
    assert out["run_id"] == "R001_fidip-w48_released" and out["split"] == "test100"
    assert out["gt"]["sha256"] and out["pred"]["sha256"]
    for k in ("metrics", "metrics_fixed", "metrics_original_scores", "ap_per_oks_thr", "per_keypoint"):
        assert k in out
    per = json.loads(per_p.read_text())
    assert len(per["records"]) == 30 and per["meta"]["run_id"] == "R001_fidip-w48_released"
    csv_p = tmp_path / "per_image.csv"
    assert ev.main(["--gt", str(gp), "--pred", str(pp), "--out", str(out_p), "--per-image", str(csv_p),
                    "--quiet"]) == 0
    assert len(csv_p.read_text().strip().splitlines()) == 31


# ------------------------------------------------------------------ paper-protocol ceiling (Test100 has ann id 0)
@pytest.mark.parametrize("rank", [1, 2, 49, 100])
def test_paper_protocol_ceiling_for_perfect_model(rank):
    """Perfect keypoints on 100 images, annotation ids 0..99. The id-0 detection is an FP at score
    rank r, so every per-threshold AP equals (r + 0.99 (100 - r)) / 101 for r >= 2 (0.99*100/101
    for r = 1), and AR = 0.99. See notebook/analysis.md."""
    gt = make_gt(100, seed=0, first_ann_id=0)
    order = [i for i in range(1, 100)]
    order.insert(rank - 1, 0)                      # image 0 at score rank `rank`
    level = {img: 0.99 - 0.007 * k for k, img in enumerate(order)}
    conf = [np.full(17, level[a["image_id"]]) for a in gt["annotations"]]
    out, _ = _m(gt, gt_as_preds(gt, conf=conf))
    expect = (0.99 * 100 / 101) if rank == 1 else (rank + 0.99 * (100 - rank)) / 101
    assert out["metrics"]["AP"] == pytest.approx(expect, abs=1e-12)
    assert out["metrics"]["AP50"] == pytest.approx(expect, abs=1e-12)
    assert out["metrics"]["AR"] == pytest.approx(0.99, abs=1e-12)
    assert out["metrics_fixed"]["AP"] == pytest.approx(1.0)
