"""Tests for the statistical-parameter configuration layer (PLAN.md sections 4 and 7)."""

from __future__ import annotations

import pytest

from funnel_sim.simulation.config import (
    ParamSpec,
    SimulationConfig,
    SimulationConfigError,
    default_config,
)
from funnel_sim.simulation.distributions import DistributionConfigError, Family
from funnel_sim.simulation.entities import Industry
from funnel_sim.simulation.effects import Effect


def test_default_config_is_valid_out_of_the_box():
    config = default_config()
    config.validate()  # should not raise
    assert config.lead_arrival.family is Family.POISSON
    assert config.deal_size.family is Family.LOGNORMAL
    assert config.industry_mix[Industry.SAAS] == pytest.approx(1.0 / len(Industry))


def test_param_spec_resolve_matches_distributions_module():
    spec = ParamSpec(Family.GAMMA, mean=3.0, variance=1.5)
    result = spec.resolve()
    assert result.dist.mean() == pytest.approx(3.0)
    assert result.dist.var() == pytest.approx(1.5)


def test_param_spec_resolve_raises_distribution_config_error_for_bad_domain():
    spec = ParamSpec(Family.BETA, mean=1.5, variance=0.1)
    with pytest.raises(DistributionConfigError):
        spec.resolve()


def test_validate_rejects_non_count_family_for_lead_arrival():
    config = default_config()
    bad = config.with_lead_arrival(Family.NORMAL, mean=20.0, variance=5.0)
    with pytest.raises(SimulationConfigError, match="lead_arrival must use a count distribution"):
        bad.validate()


def test_validate_rejects_non_positive_continuous_family_for_deal_size():
    config = default_config()
    bad = config.with_deal_size(Family.POISSON, mean=5.0, variance=None)
    with pytest.raises(SimulationConfigError, match="deal_size must use a positive continuous"):
        bad.validate()


@pytest.mark.parametrize("bad_prob", [0.0, 1.0, -0.1, 1.5])
def test_validate_rejects_base_close_prob_out_of_open_interval(bad_prob):
    config = default_config()
    config.base_close_prob = bad_prob
    with pytest.raises(SimulationConfigError, match="base_close_prob"):
        config.validate()


def test_validate_rejects_industry_mix_that_does_not_sum_to_one():
    config = default_config()
    config.industry_mix = {Industry.SAAS: 0.5, Industry.RETAIL: 0.6}
    with pytest.raises(SimulationConfigError, match="must sum to 1.0"):
        config.validate()


def test_validate_rejects_negative_industry_mix_weight():
    config = default_config()
    config.industry_mix = {Industry.SAAS: 1.2, Industry.RETAIL: -0.2}
    with pytest.raises(SimulationConfigError, match="must all be >= 0"):
        config.validate()


def test_validate_rejects_empty_industry_mix():
    config = default_config()
    config.industry_mix = {}
    with pytest.raises(SimulationConfigError, match="must not be empty"):
        config.validate()


def test_validate_surfaces_bad_lead_arrival_distribution_config():
    config = default_config()
    config.lead_arrival = ParamSpec(Family.NEGATIVE_BINOMIAL, mean=10.0, variance=5.0)  # var < mean
    with pytest.raises(DistributionConfigError, match="variance must be strictly greater than mean"):
        config.validate()


def test_with_lead_arrival_and_with_deal_size_return_new_configs_without_mutating():
    config = default_config()
    updated = config.with_lead_arrival(Family.NEGATIVE_BINOMIAL, mean=15.0, variance=40.0)

    assert updated.lead_arrival.family is Family.NEGATIVE_BINOMIAL
    assert config.lead_arrival.family is Family.POISSON  # original untouched

    updated2 = config.with_deal_size(Family.GAMMA, mean=6000.0, variance=2_000_000.0)
    assert updated2.deal_size.family is Family.GAMMA
    assert config.deal_size.family is Family.LOGNORMAL


def test_industry_and_rep_skill_effects_are_plain_effect_instances():
    config = SimulationConfig(
        seed=1,
        lead_arrival=ParamSpec(Family.POISSON, mean=10.0),
        deal_size=ParamSpec(Family.LOGNORMAL, mean=5000.0, variance=1_000_000.0),
        industry_effects={Industry.SAAS: Effect(name="industry_saas", logit_delta=0.3)},
        rep_skill_effects={"rep-1": Effect(name="rep_1_skill", logit_delta=-0.2)},
    )
    assert config.industry_effects[Industry.SAAS].logit_delta == pytest.approx(0.3)
    assert config.rep_skill_effects["rep-1"].logit_delta == pytest.approx(-0.2)
