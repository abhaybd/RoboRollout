import pytest
from omegaconf import OmegaConf

from robo_rollout.config import ImageInputConfig, RolloutConfig
from robo_rollout.rollout import TimingBudget, check_config


def make_cfg(**overrides):
    cfg = OmegaConf.structured(RolloutConfig(policy_dt=0.1, task="pick", image_input=ImageInputConfig(width=64, height=48)))
    return OmegaConf.merge(cfg, overrides)


def test_check_config_accepts_complete_config():
    check_config(make_cfg(num_trials=0))


def test_check_config_reports_missing_keys():
    cfg = make_cfg(policy_cameras={"exo": "???"})
    with pytest.raises(ValueError, match="policy_cameras.exo"):
        check_config(cfg)


def test_check_config_rejects_negative_num_trials():
    with pytest.raises(AssertionError, match="num_trials"):
        check_config(make_cfg(num_trials=-1))


def test_timing_budget_splits_remainder_among_unspecified():
    budget = TimingBudget(obs=0.01, act=0.03).resolve(policy_dt=0.1)
    assert budget.obs == 0.01 and budget.act == 0.03
    assert budget.infer == pytest.approx(0.06)

    budget = TimingBudget(act=0.02).resolve(policy_dt=0.1)
    assert budget.obs == pytest.approx(0.04) and budget.infer == pytest.approx(0.04)


def test_timing_budget_remainder_is_never_negative():
    budget = TimingBudget(obs=0.08, act=0.08).resolve(policy_dt=0.1)
    assert budget.infer == 0.0
