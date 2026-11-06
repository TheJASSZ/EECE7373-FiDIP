"""Tests for eval/analyze.py: exactness of the id-0 rank curve and an end-to-end figure smoke test."""
import copy
import json
import os

import numpy as np
import pytest

from conftest import make_gt, make_preds
import analyze as an
import eval_coco as ev


def test_id0_rank_curve_matches_evaluator_at_actual_rank(tmp_path):
    gt = make_gt(40, seed=3, first_ann_id=0)
    preds = make_preds(gt, 0.02, seed=4)
    p = tmp_path / "p.json"
    p.write_text(json.dumps(preds))
    G = ev.GroundTruth(copy.deepcopy(gt))
    rows, actual, n = an.id0_rank_curve(G, str(p))
    out, _ = ev.evaluate(copy.deepcopy(gt), preds)
    assert n == 40 and rows[actual - 1]["actual"]
    assert out["id0_artifact"]["score_rank"] == actual
    assert rows[actual - 1]["AP_paper"] == pytest.approx(out["metrics"]["AP"], abs=1e-12)
    assert rows[actual - 1]["AP_fixed"] == pytest.approx(out["metrics_fixed"]["AP"], abs=1e-12)
    ap = np.array([r["AP_paper"] for r in rows])
    assert ap[-1] >= ap[0]                         # an FP ranked last costs least


def test_models_and_splits_figures_smoke(tmp_path):
    gt_t, gt_v = make_gt(30, seed=1, first_ann_id=0, n_v1=2), make_gt(40, seed=2, start_image_id=100, first_ann_id=100)
    runs = {}
    for name, nz in (("A", 0.02), ("B", 0.03)):
        d = tmp_path / name
        d.mkdir()
        for split, g in (("test100", gt_t), ("val400", gt_v)):
            pp = d / f"preds_{split}.json"
            pp.write_text(json.dumps(make_preds(g, nz, seed=7)))
            out, recs = ev.evaluate(copy.deepcopy(g), str(pp))
            (d / f"metrics_{split}.json").write_text(json.dumps(out))
            (d / f"per_image_{split}.json").write_text(json.dumps({"meta": {}, "records": recs}))
        runs[name] = d
    gpath = tmp_path / "test100.json"
    gpath.write_text(json.dumps(gt_t))
    man = {"split": "test100", "gt": str(gpath), "focus": ["A", "B"], "pairs": [["A", "B"]],
           "published": {"AP": 0.936, "AP50": 0.985, "AP75": 0.985, "AR": 0.946},
           "models": [{"key": k, "label": k, "slot": i + 1,
                       "runs": [{"run_id": f"R00{i}_{k}", "per_image": str(runs[k] / "per_image_test100.json")}]}
                      for i, k in enumerate(("A", "B"))]}
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(man))
    fig, tab = tmp_path / "fig", tmp_path / "tab"
    an.main(["models", "--manifest", str(mp), "--outdir", str(fig), "--tables", str(tab)])
    for stem in ("ap_vs_oks_threshold", "oks_distribution", "per_keypoint_error", "failure_taxonomy",
                 "error_vs_orientation", "score_vs_oks", "oks_scatter_A_vs_B"):
        assert (fig / f"{stem}_test100.pdf").exists() and (fig / f"{stem}_test100.png").exists(), stem
    for t in ("ap_per_threshold", "per_keypoint", "failure_taxonomy", "lr_swaps", "occlusion", "ranking_quality",
              "per_image_taxonomy", "delta_by_category_A_vs_B"):
        assert (tab / f"{t}_test100.csv").exists(), t
    rq = list(__import__("csv").DictReader(open(tab / "ranking_quality_test100.csv")))
    assert all(float(r["AP_oracle_fixed"]) >= float(r["AP_fixed"]) - 1e-9 for r in rq)   # oracle ranking is an upper bound
    an.main(["splits", "--label", "A", "--name", "A", "--a", f"test100={runs['A'] / 'per_image_test100.json'}",
             "--b", f"val400={runs['A'] / 'per_image_val400.json'}", "--outdir", str(fig), "--tables", str(tab)])
    assert (fig / "split_difficulty_A.png").exists() and (tab / "split_difficulty_A.csv").exists()
