"""Tests for the statistical-parameter configuration layer (PLAN.md sections 4 and 7)."""

from __future__ import annotations

import pytest

from funnel_sim.simulation.config import (
    COUNT_FAMILIES,
    DAYS_UNTIL_CONVERTED_FAMILIES,
    DEAL_SIZE_FAMILIES,
    PROBABILITY_FAMILIES,
    ParamSpec,
    SimulationConfig,
    SimulationConfigError,
    default_config,
    validate_deal_size_spec,
)
from funnel_sim.simulation.distributions import DistributionConfigError, Family
from funnel_sim.simulation.entities import Industry
from funnel_sim.simulation.effects import Effect


def test_default_config_is_valid_out_of_the_box():
    config = default_config()
    config.validate()  # should not raise
    assert config.lead_arrival.family is Family.POISSON
    assert config.deal_size.family is Family.NORMAL  # most intuitive default (Phase 1 feedback)
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


def test_validate_rejects_non_deal_size_family_for_deal_size():
    config = default_config()
    bad = config.with_deal_size(Family.POISSON, mean=5.0, variance=None)
    with pytest.raises(SimulationConfigError, match="deal_size must use one of"):
        bad.validate()


def test_deal_size_families_include_normal_gamma_lognormal_and_continuous_uniform():
    assert set(DEAL_SIZE_FAMILIES) == {
        Family.NORMAL,
        Family.GAMMA,
        Family.LOGNORMAL,
        Family.CONTINUOUS_UNIFORM,
    }


def test_count_families_include_poisson_negative_binomial_and_discrete_uniform():
    assert set(COUNT_FAMILIES) == {
        Family.POISSON,
        Family.NEGATIVE_BINOMIAL,
        Family.DISCRETE_UNIFORM,
    }


def test_param_spec_resolve_handles_discrete_uniform_from_low_and_high():
    spec = ParamSpec(Family.DISCRETE_UNIFORM, low=0, high=100)
    result = spec.resolve()
    assert result.dist.mean() == pytest.approx(50.0)


def test_param_spec_resolve_handles_continuous_uniform_from_low_and_high():
    spec = ParamSpec(Family.CONTINUOUS_UNIFORM, low=50.0, high=10_000.0)
    result = spec.resolve()
    assert result.dist.mean() == pytest.approx(5_025.0)


def test_param_spec_resolve_rejects_uniform_with_high_not_greater_than_low():
    spec = ParamSpec(Family.CONTINUOUS_UNIFORM, low=100.0, high=100.0)
    with pytest.raises(DistributionConfigError, match="must be greater than low"):
        spec.resolve()


def test_validate_deal_size_spec_rejects_negative_normal_mean():
    with pytest.raises(SimulationConfigError, match=r"mean of \$0 or more"):
        validate_deal_size_spec(ParamSpec(Family.NORMAL, mean=-100.0, variance=1000.0))


def test_validate_deal_size_spec_accepts_zero_or_positive_normal_mean():
    validate_deal_size_spec(ParamSpec(Family.NORMAL, mean=0.0, variance=1000.0))  # no raise
    validate_deal_size_spec(ParamSpec(Family.NORMAL, mean=5000.0, variance=1000.0))  # no raise


def test_validate_deal_size_spec_ignores_mean_sign_for_non_normal_families():
    # Gamma/Lognormal already require mean > 0 at the distributions.py layer (resolve() would
    # catch a non-positive mean there); validate_deal_size_spec itself only special-cases Normal.
    validate_deal_size_spec(ParamSpec(Family.GAMMA, mean=100.0, variance=50.0))  # no raise


def test_validate_config_with_normal_deal_size_and_negative_mean_raises():
    config = default_config().with_deal_size(Family.NORMAL, mean=-50.0, variance=100.0)
    with pytest.raises(SimulationConfigError, match=r"mean of \$0 or more"):
        config.validate()


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
    assert config.deal_size.family is Family.NORMAL  # original untouched


def test_default_config_has_reasonable_conversion_defaults():
    config = default_config()
    config.validate()  # should not raise
    assert config.days_until_converted.family is Family.POISSON
    assert config.days_until_converted.mean == pytest.approx(5.0)
    assert config.base_conversion_prob.family is Family.BETA
    assert config.base_conversion_prob.mean == pytest.approx(0.20)
    assert config.conversion_decay.family is Family.BETA
    assert config.conversion_decay.mean == pytest.approx(0.8)


def test_days_until_converted_families_are_poisson_negative_binomial_and_discrete_uniform():
    assert set(DAYS_UNTIL_CONVERTED_FAMILIES) == {
        Family.POISSON,
        Family.NEGATIVE_BINOMIAL,
        Family.DISCRETE_UNIFORM,
    }


def test_probability_families_are_beta_and_continuous_uniform():
    assert set(PROBABILITY_FAMILIES) == {Family.BETA, Family.CONTINUOUS_UNIFORM}


def test_validate_rejects_non_count_family_for_days_until_converted():
    config = default_config()
    config.days_until_converted = ParamSpec(Family.NORMAL, mean=5.0, variance=4.0)
    with pytest.raises(SimulationConfigError, match="days_until_converted must use one of"):
        config.validate()


def test_validate_accepts_continuous_uniform_for_base_conversion_prob_and_conversion_decay():
    config = default_config()
    config.base_conversion_prob = ParamSpec(Family.CONTINUOUS_UNIFORM, low=0.1, high=0.3)
    config.conversion_decay = ParamSpec(Family.CONTINUOUS_UNIFORM, low=0.7, high=0.9)
    config.validate()  # should not raise


def test_validate_rejects_non_probability_family_for_base_conversion_prob():
    config = default_config()
    config.base_conversion_prob = ParamSpec(Family.NORMAL, mean=0.2, variance=0.01)
    with pytest.raises(SimulationConfigError, match="base_conversion_prob must use one of"):
        config.validate()


def test_validate_rejects_non_probability_family_for_conversion_decay():
    config = default_config()
    config.conversion_decay = ParamSpec(Family.NORMAL, mean=0.8, variance=0.01)
    with pytest.raises(SimulationConfigError, match="conversion_decay must use one of"):
        config.validate()


def test_validate_surfaces_bad_base_conversion_prob_distribution_config():
    config = default_config()
    # Beta variance must stay below mean*(1-mean) -- 0.5 is far too large for mean=0.2.
    config.base_conversion_prob = ParamSpec(Family.BETA, mean=0.2, variance=0.5)
    with pytest.raises(DistributionConfigError, match="variance must be between"):
        config.validate()


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
