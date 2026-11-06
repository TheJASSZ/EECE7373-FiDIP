"""The Test100 gate in scripts/analysis_pipeline.py: only reproduction targets and pre-registered runs."""
import os
import sys

from conftest import REPO

sys.path.insert(0, os.path.join(REPO, "scripts"))
import analysis_pipeline as apl  # noqa: E402


def test_gate():
    reg = {"R016_fidip-w48_lam0.001_s1": {"variant": "lambda sweep"}, "R010_fidip-w48_retrain_s1": {"variant": "retrain"}}
    pre = {"R030_vitpose-h_s1"}
    assert apl.test_allowed("R001_fidip-w48_released", reg, pre)[0]
    assert apl.test_allowed("R010_fidip-w48_retrain_s1", reg, pre)[0]
    assert apl.test_allowed("R030_vitpose-h_s1", reg, pre)[0]
    assert not apl.test_allowed("R016_fidip-w48_lam0.001_s1", reg, pre)[0]
    assert not apl.test_allowed("R031_vitpose-h_s1", reg, pre)[0]
    assert not apl.test_allowed("R040_mmpose-hrnet-w48_retrainlike_s1", {}, set())[0]


def test_validate500_is_gated_like_test100():
    assert "validate500" in apl.TEST_LIKE and "test100" in apl.TEST_LIKE and "val400" not in apl.TEST_LIKE
