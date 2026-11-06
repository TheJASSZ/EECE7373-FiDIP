"""Smoke test for scripts/collect_results.py: provenance, Test100 withholding, LaTeX macros."""
import json
import os
import sys

from conftest import REPO, make_gt, make_preds
import eval_coco as ev

sys.path.insert(0, os.path.join(REPO, "scripts"))
import collect_results as cr  # noqa: E402


def _setup(tmp_path):
    gt_t, gt_v = make_gt(20, seed=1, first_ann_id=0), make_gt(30, seed=2, start_image_id=100, first_ann_id=100)
    ev_root = tmp_path / "eval"
    runs = {"R001_fidip-w48_released": 0.02, "R020_mmpose-hrnet-w48_s1": 0.03, "R016_fidip-w48_lam0.01_s1": 0.025}
    for rid, nz in runs.items():
        d = ev_root / rid
        d.mkdir(parents=True)
        for split, g in (("test100", gt_t), ("val400", gt_v)):
            out, _ = ev.evaluate(json.loads(json.dumps(g)), make_preds(g, nz, seed=hash(rid) % 97))
            (d / f"metrics_{split}.json").write_text(json.dumps(out))
    (tmp_path / "registry.csv").write_text(
        "run_id,job_id,stream,model,variant,seed,status,notes\n"
        "R001_fidip-w48_released,1,repro,fidip-w48,released,,done,\n"
        "R020_mmpose-hrnet-w48_s1,2,improve,mmpose-hrnet-w48,control,1,done,\n"
        "R016_fidip-w48_lam0.01_s1,3,repro,fidip-w48,lambda 0.01,1,done,\n")
    (tmp_path / "prereg.md").write_text("| # | at | runs |\n|---|---|---|\n")
    (tmp_path / "configs.json").write_text(json.dumps({"configs": {
        "released": {"label": "FiDIP (released)", "role": "released", "macro": "Released",
                     "run_ids": ["R001_fidip-w48_released"]}}}))
    return tmp_path


def test_collect_withholds_unregistered_test100(tmp_path):
    t = _setup(tmp_path)
    tex = t / "gen" / "results_tables.tex"
    args = ["--registry", str(t / "registry.csv"), "--prereg", str(t / "prereg.md"), "--configs",
            str(t / "configs.json"), "--eval-root", str(t / "eval"), "--runs-root", str(t / "none"),
            "--tables", str(t / "tables"), "--tex", str(tex)]
    assert cr.main(args) == 0
    attempts = json.load(open(t / "tables" / "attempts.json"))
    by = {a["config"]: a for a in attempts}
    assert by["released"]["test100_reported"] is True and by["released"]["test100_AP"] is not None
    others = [a for k, a in by.items() if k != "released"]
    assert others and all(a["test100_AP"] is None and not a["test100_reported"] for a in others)
    assert all(a["val400_AP"] is not None for a in attempts)
    per_run = json.load(open(t / "tables" / "per_run.json"))
    for r in per_run:
        if r["run_id"] != "R001_fidip-w48_released":
            assert "test100_AP" not in r and r.get("test100_note", "").startswith("withheld")
    s = tex.read_text()
    assert "\\newcommand{\\ReleasedTestAP}" in s and "\\newcommand{\\PaperFiDIPAP}{93.6}" in s
    assert "R001_fidip-w48_released" in s                     # provenance comments
    names = [ln.split("}")[0] for ln in s.splitlines() if ln.startswith("\\newcommand")]
    assert len(names) == len(set(names))                       # no duplicate macros
    assert (t / "gen" / "MACROS.md").exists()


def test_collect_includes_tagged_val400_ablations(tmp_path):
    t = _setup(tmp_path)
    src = t / "eval" / "R020_mmpose-hrnet-w48_s1" / "metrics_val400.json"
    (t / "eval" / "R020_mmpose-hrnet-w48_s1" / "metrics_val400_noflip.json").write_text(src.read_text())
    (t / "eval" / "R020_mmpose-hrnet-w48_s1" / "metrics_test100_noflip.json").write_text(src.read_text())
    args = ["--registry", str(t / "registry.csv"), "--prereg", str(t / "prereg.md"), "--configs",
            str(t / "configs.json"), "--eval-root", str(t / "eval"), "--runs-root", str(t / "none"),
            "--tables", str(t / "tables"), "--tex", str(t / "gen" / "x.tex")]
    assert cr.main(args) == 0
    rows = [a for a in json.load(open(t / "tables" / "attempts.json")) if a["role"] == "test-time ablation"]
    assert len(rows) == 1 and rows[0]["config"].endswith("_noflip") and rows[0]["test100_AP"] is None
