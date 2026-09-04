"""Tests for the distribution parameterization layer (PLAN.md section 4).

Correctness is checked against the frozen scipy distribution's own `.mean()`/`.var()` rather
than by sampling — that's an exact, deterministic check of the method-of-moments algebra instead
of a statistical (and potentially flaky) one. A couple of sampling-based tests exist too, to
sanity-check that `.rvs()` actually behaves like the distribution it claims to be.
"""

from __future__ import annotations

import numpy as np
import pytest

from funnel_sim.simulation.distributions import (
    DistributionConfigError,
    Family,
    moments_to_params,
    preview_xy,
    sample_positive,
    uniform_from_bounds,
)


@pytest.mark.parametrize(
    "family, mean, variance",
    [
        (Family.NORMAL, 10.0, 4.0),
        (Family.NORMAL, -5.0, 2.5),
        (Family.GAMMA, 3.0, 1.5),
        (Family.BETA, 0.3, 0.02),
        (Family.LOGNORMAL, 5.0, 3.0),
        (Family.NEGATIVE_BINOMIAL, 4.0, 10.0),
    ],
)
def test_recovers_configured_mean_and_variance(family, mean, variance):
    result = moments_to_params(family, mean, variance)
    assert result.dist.mean() == pytest.approx(mean, rel=1e-6, abs=1e-9)
    assert result.dist.var() == pytest.approx(variance, rel=1e-6, abs=1e-9)
    assert result.warnings == ()


def test_poisson_forces_variance_to_equal_mean_and_warns():
    with pytest.warns(UserWarning, match="Poisson forces variance == mean"):
        result = moments_to_params(Family.POISSON, mean=6.0, variance=20.0)
    assert result.dist.mean() == pytest.approx(6.0)
    assert result.dist.var() == pytest.approx(6.0)
    assert len(result.warnings) == 1
    assert "Negative Binomial" in result.warnings[0]


def test_poisson_with_matching_variance_does_not_warn(recwarn):
    result = moments_to_params(Family.POISSON, mean=6.0, variance=6.0)
    assert result.warnings == ()
    assert len(recwarn) == 0


def test_poisson_with_no_variance_given_does_not_warn(recwarn):
    result = moments_to_params(Family.POISSON, mean=6.0, variance=None)
    assert result.warnings == ()
    assert len(recwarn) == 0


@pytest.mark.parametrize(
    "family, mean, variance, match",
    [
        (Family.NORMAL, 1.0, 0.0, "variance must be > 0"),
        (Family.NORMAL, 1.0, -1.0, "variance must be > 0"),
        (Family.GAMMA, 0.0, 1.0, "mean must be > 0"),
        (Family.GAMMA, 1.0, 0.0, "variance must be > 0"),
        (Family.BETA, 0.0, 0.01, "strictly between 0 and 1"),
        (Family.BETA, 1.0, 0.01, "strictly between 0 and 1"),
        (Family.BETA, 0.5, 0.3, "variance must be between"),
        (Family.POISSON, 0.0, None, "mean must be > 0"),
        (Family.LOGNORMAL, -1.0, 1.0, "mean must be > 0"),
        (Family.LOGNORMAL, 1.0, 0.0, "variance must be > 0"),
        (Family.NEGATIVE_BINOMIAL, 5.0, 5.0, "variance must be strictly greater than mean"),
        (Family.NEGATIVE_BINOMIAL, 5.0, 3.0, "variance must be strictly greater than mean"),
    ],
)
def test_out_of_domain_configs_raise_clear_errors(family, mean, variance, match):
    with pytest.raises(DistributionConfigError, match=match):
        moments_to_params(family, mean, variance)


def test_sampling_matches_configured_moments_for_gamma():
    result = moments_to_params(Family.GAMMA, mean=3.0, variance=1.5)
    rng = np.random.default_rng(0)
    samples = result.dist.rvs(size=200_000, random_state=rng)
    assert samples.mean() == pytest.approx(3.0, rel=0.02)
    assert samples.var() == pytest.approx(1.5, rel=0.05)


def test_sampling_matches_configured_moments_for_negative_binomial():
    result = moments_to_params(Family.NEGATIVE_BINOMIAL, mean=4.0, variance=10.0)
    rng = np.random.default_rng(0)
    samples = result.dist.rvs(size=200_000, random_state=rng)
    assert samples.mean() == pytest.approx(4.0, rel=0.02)
    assert samples.var() == pytest.approx(10.0, rel=0.05)


@pytest.mark.parametrize(
    "family, mean, variance",
    [
        (Family.NORMAL, 10.0, 4.0),
        (Family.GAMMA, 3.0, 1.5),
        (Family.BETA, 0.3, 0.02),
        (Family.LOGNORMAL, 5.0, 3.0),
    ],
)
def test_continuous_preview_pdf_integrates_to_roughly_one(family, mean, variance):
    result = moments_to_params(family, mean, variance)
    x, y = preview_xy(result)
    assert x.shape == y.shape
    assert np.all(y >= 0)
    # Rough numerical integration of the previewed pdf slice — not exactly 1 since the preview
    # deliberately covers the 0.1st-99.9th percentile range, not the full support.
    area = np.trapezoid(y, x)
    assert area == pytest.approx(0.998, abs=0.02)


@pytest.mark.parametrize(
    "family, mean, variance",
    [
        (Family.POISSON, 6.0, 6.0),
        (Family.NEGATIVE_BINOMIAL, 4.0, 10.0),
    ],
)
def test_discrete_preview_pmf_sums_to_roughly_one(family, mean, variance):
    result = moments_to_params(family, mean, variance)
    x, y = preview_xy(result)
    assert x.shape == y.shape
    assert np.all(x == np.round(x))  # integer support
    assert np.all(y >= 0)
    assert y.sum() == pytest.approx(0.999, abs=0.01)


def test_preview_xy_with_lower_bound_only_covers_x_above_the_bound():
    result = moments_to_params(Family.NORMAL, mean=100.0, variance=50.0**2)
    x, y = preview_xy(result, lower_bound=0.0)
    assert x.min() >= 0.0
    assert np.all(y >= 0)
    # The truncated density is renormalized -- it should integrate to roughly 1 over the shown
    # range, same as the untruncated preview does (not to Pr(X > 0), which is < 1).
    area = np.trapezoid(y, x)
    assert area == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize(
    "family, mean, variance",
    [
        (Family.GAMMA, 3.0, 1.5),
        (Family.LOGNORMAL, 5.0, 3.0),
    ],
)
def test_preview_xy_with_lower_bound_zero_is_unchanged_for_already_positive_families(family, mean, variance):
    # Gamma/Lognormal already have Pr(X > 0) ~= 1, so truncating at 0 shouldn't meaningfully
    # reshape the curve.
    result = moments_to_params(family, mean, variance)
    x_plain, y_plain = preview_xy(result)
    x_truncated, y_truncated = preview_xy(result, lower_bound=0.0)
    assert x_truncated[0] == pytest.approx(0.0, abs=1e-6)
    assert x_plain[-1] == pytest.approx(x_truncated[-1], rel=0.01)
    assert y_plain[-1] == pytest.approx(y_truncated[-1], rel=0.01)


def test_preview_xy_raises_when_almost_nothing_is_above_the_bound():
    # Mean far below the bound -- virtually none of the mass is above it.
    result = moments_to_params(Family.NORMAL, mean=-1000.0, variance=10.0)
    with pytest.raises(DistributionConfigError, match="nothing meaningful to preview"):
        preview_xy(result, lower_bound=0.0)


def test_sample_positive_only_returns_values_above_the_bound():
    result = moments_to_params(Family.NORMAL, mean=100.0, variance=50.0**2)
    rng = np.random.default_rng(0)
    samples = sample_positive(result, size=5_000, rng=rng, lower_bound=0.0)
    assert len(samples) == 5_000
    assert np.all(samples > 0.0)


def test_sample_positive_matches_the_truncated_distributions_moments():
    # Pr(X > 0) is comfortably > 0.5 here, so this should converge in very few rounds and the
    # accepted values should look like draws from the analytically truncated Normal.
    result = moments_to_params(Family.NORMAL, mean=8_000.0, variance=3_000.0**2)
    rng = np.random.default_rng(1)
    samples = sample_positive(result, size=50_000, rng=rng, lower_bound=0.0)
    assert np.all(samples > 0.0)
    assert samples.mean() == pytest.approx(result.dist.mean(), rel=0.05)


def test_uniform_from_bounds_discrete_matches_low_and_high():
    result = uniform_from_bounds(Family.DISCRETE_UNIFORM, low=0, high=100)
    assert result.dist.mean() == pytest.approx(50.0)
    rng = np.random.default_rng(0)
    samples = result.dist.rvs(size=10_000, random_state=rng)
    assert samples.min() >= 0
    assert samples.max() <= 100
    assert np.all(samples == np.round(samples))


def test_uniform_from_bounds_continuous_matches_low_and_high():
    result = uniform_from_bounds(Family.CONTINUOUS_UNIFORM, low=50.0, high=10_000.0)
    assert result.dist.mean() == pytest.approx(5_025.0)
    rng = np.random.default_rng(0)
    samples = result.dist.rvs(size=10_000, random_state=rng)
    assert samples.min() >= 50.0
    assert samples.max() <= 10_000.0


def test_uniform_from_bounds_rejects_high_not_greater_than_low():
    with pytest.raises(DistributionConfigError, match="must be greater than low"):
        uniform_from_bounds(Family.CONTINUOUS_UNIFORM, low=100.0, high=50.0)
    with pytest.raises(DistributionConfigError, match="must be greater than low"):
        uniform_from_bounds(Family.DISCRETE_UNIFORM, low=10, high=10)


def test_uniform_from_bounds_preview_and_sample_positive_work_like_other_families():
    result = uniform_from_bounds(Family.CONTINUOUS_UNIFORM, low=50.0, high=10_000.0)
    x, y = preview_xy(result)
    assert x.shape == y.shape
    assert np.all(y >= 0)

    rng = np.random.default_rng(0)
    samples = sample_positive(result, size=1_000, rng=rng, lower_bound=0.0)
    assert len(samples) == 1_000
    assert np.all(samples > 0.0)


def test_sample_positive_raises_when_it_cannot_converge():
    # Mean far below the bound -- essentially nothing to accept, so max_rounds is exhausted.
    result = moments_to_params(Family.NORMAL, mean=-1000.0, variance=10.0)
    rng = np.random.default_rng(0)
    with pytest.raises(DistributionConfigError, match="Could not draw"):
        sample_positive(result, size=10, rng=rng, lower_bound=0.0, max_rounds=3)
