"""The Test100 gate in scripts/analysis_pipeline.py: only reproduction targets and pre-registered runs."""
import os
import sys

from conftest import REPO

sys.path.insert(0, os.path.join(REPO, "scripts"))
import analysis_pipeline as apl  # noqa: E402


def test_gate():
    reg = {"R016_fidip-w48_lam0.001_s1": {"variant": "lambda sweep"}, "R010_fidip-w48_retrain_s1": {"variant": "retrain"}}
    pre = {"R030_vitpose-h_s1"}
    kw = {"roles": {}}
    assert apl.test_allowed("R001_fidip-w48_released", reg, pre, **kw)[0]
    assert apl.test_allowed("R010_fidip-w48_retrain_s1", reg, pre, **kw)[0]
    assert apl.test_allowed("R030_vitpose-h_s1", reg, pre, **kw)[0]
    assert not apl.test_allowed("R016_fidip-w48_lam0.001_s1", reg, pre, **kw)[0]
    assert not apl.test_allowed("R031_vitpose-h_s1", reg, pre, **kw)[0]
    assert not apl.test_allowed("R040_mmpose-hrnet-w48_retrainlike_s1", {}, set(), **kw)[0]


def test_gate_configs_roles_are_authoritative():
    roles = {"R012_fidip-w48_retrain-lam0_s1": "retrain", "R017_fidip-w48_retrain_sweep_s1": "ablation"}
    assert apl.test_allowed("R012_fidip-w48_retrain-lam0_s1", {}, set(), roles=roles)[0]
    assert not apl.test_allowed("R017_fidip-w48_retrain_sweep_s1", {}, set(), roles=roles)[0]   # name says retrain


def test_validate500_is_gated_like_test100():
    assert "validate500" in apl.TEST_LIKE and "test100" in apl.TEST_LIKE and "val400" not in apl.TEST_LIKE


def test_split_and_tag_parsing():
    m = apl.SPLIT_RE.match("preds_val400_noflip.json")
    assert m and m.group(1) == "val400" and m.group(2) == "_noflip"
    m = apl.SPLIT_RE.match("preds_test100.json")
    assert m and m.group(1) == "test100" and m.group(2) == ""
    assert apl.SPLIT_RE.match("preds_validate500.json").group(1) == "validate500"
    assert apl.SPLIT_RE.match("preds_train_infant.json") is None


def test_validate_infant_alias_is_gated_and_scored_on_test100():
    assert apl.SPLIT_RE.match("preds_validate_infant.json").group(1) == "validate_infant"
    assert apl.SPLIT_ALIAS_GT["validate_infant"] == "test100" and "validate_infant" in apl.TEST_LIKE


def test_prereg_parser_ranges_and_forbidden_rows(tmp_path):
    import prereg
    p = tmp_path / "PREREG.md"
    p.write_text("| # | at | runs | evidence | rule | result |\n|---|---|---|---|---|---|\n"
                 "| repro-1 | x | R010-R012 `a_s{1,2,3}` and R013–R015 `b` | reproduction | best | |\n"
                 "| repro-2 | x | R016-R018 sweep; R019 paper recipe; R020 labels | exploratory | **Val400 only, no Test100** | n/a |\n"
                 "| imp-1 | x | R102_mmp-vitpose-l256_base_s1, R105 | Val400 95 | best | |\n")
    pre = prereg.allowed_prefixes(str(p))
    assert {"R010", "R011", "R012", "R013", "R014", "R015", "R102", "R105"} <= pre
    assert not ({"R016", "R017", "R018", "R019", "R020"} & pre)
    assert prereg.is_registered("R011_fidip-w48_lambda5e-4_s2", pre)
    assert not prereg.is_registered("R017_fidip-w48_lam2e-3_s1", pre)
    assert apl.test_allowed("R014_fidip-w48_lambda0_s2", {}, pre, roles={})[0]
    assert not apl.test_allowed("R019_fidip-w48_paper_s1", {}, pre, roles={})[0]
