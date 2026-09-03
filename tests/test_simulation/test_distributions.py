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
