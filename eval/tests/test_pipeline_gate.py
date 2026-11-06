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
