"""Distribution parameterization for the stats-configuration layer.

Users configure a random variable the way they naturally think about one — pick a family, give
a mean and a variance — rather than in the family's native (and often unintuitive) parameters.
This module converts mean/variance into native parameters via method-of-moments, validates the
result against each family's valid domain, and exposes preview data (x/y arrays) for the
distribution-picker widget described in PLAN.md section 7.

See PLAN.md section 4 for the design rationale, in particular the Poisson mean == variance
constraint and why Negative Binomial exists as its over-dispersed counterpart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import warnings

import numpy as np
from scipy import stats


class DistributionConfigError(ValueError):
    """Raised when a (family, mean, variance) configuration is outside that family's valid
    domain. Carries a message meant to be shown directly to the user in the UI — see PLAN.md
    section 4: "surface a clear error rather than silently producing garbage."
    """


class Family(str, Enum):
    """Distribution families offered by the stats-configuration UI (PLAN.md section 4)."""

    NORMAL = "normal"
    GAMMA = "gamma"
    BETA = "beta"
    POISSON = "poisson"
    LOGNORMAL = "lognormal"
    NEGATIVE_BINOMIAL = "negative_binomial"
    DISCRETE_UNIFORM = "discrete_uniform"
    CONTINUOUS_UNIFORM = "continuous_uniform"


# Families whose support is the non-negative integers (pmf, not pdf; discrete preview x-axis).
DISCRETE_FAMILIES = frozenset({Family.POISSON, Family.NEGATIVE_BINOMIAL, Family.DISCRETE_UNIFORM})

# Families with only one free parameter -- variance isn't a meaningful independent input (PLAN.md
# section 4: Poisson forces variance == mean). The UI hides the variance control for these.
SINGLE_PARAMETER_FAMILIES = frozenset({Family.POISSON})

# Families parameterized directly by their two endpoints rather than by mean/variance -- see
# `uniform_from_bounds` below. The UI shows a "Low"/"High" pair instead of "Mean"/"Variance" for
# these (Phase 1 feedback: "the two parameters should be the left and right endpoints").
UNIFORM_FAMILIES = frozenset({Family.DISCRETE_UNIFORM, Family.CONTINUOUS_UNIFORM})


@dataclass(frozen=True)
class DistributionResult:
    """The outcome of converting a (family, mean, variance) config into a usable distribution.

    Attributes:
        family: the distribution family that was configured.
        mean: the mean the caller asked for.
        variance: the variance the caller asked for (may differ from what was actually applied
            — see `warnings`, e.g. Poisson always ignores it).
        native_params: the family's native parameters, in the exact keyword form needed to build
            `dist` (also handy for display in the UI, e.g. "shape=2.0, scale=3.0").
        dist: a frozen scipy.stats distribution — call `.rvs(size=..., random_state=...)` to
            sample, `.pdf`/`.pmf`, `.mean()`, `.var()`, `.ppf()`, etc.
        warnings: human-readable notes about the conversion (e.g. Poisson's variance constraint).
            Non-fatal — unlike DistributionConfigError, these describe a config that was still
            honored, just not exactly as asked.
    """

    family: Family
    mean: float
    variance: float | None
    native_params: dict[str, float]
    dist: Any
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DistributionConfigError(message)


def moments_to_params(family: Family, mean: float, variance: float | None) -> DistributionResult:
    """Convert a (mean, variance) configuration into a distribution via method-of-moments.

    Raises DistributionConfigError if the requested mean/variance falls outside the family's
    valid domain (e.g. a Beta mean outside (0, 1)). Never silently clamps to a nearby valid
    value — the caller sees exactly why the configuration doesn't work.
    """
    family = Family(family)
    notes: list[str] = []

    if family is Family.NORMAL:
        _require(variance is not None and variance > 0, "Normal: variance must be > 0.")
        std = float(np.sqrt(variance))
        params = {"loc": float(mean), "scale": std}
        dist = stats.norm(**params)

    elif family is Family.GAMMA:
        _require(mean is not None and mean > 0, "Gamma: mean must be > 0.")
        _require(variance is not None and variance > 0, "Gamma: variance must be > 0.")
        shape = mean**2 / variance
        scale = variance / mean
        params = {"a": float(shape), "scale": float(scale)}
        dist = stats.gamma(**params)

    elif family is Family.BETA:
        _require(
            mean is not None and 0 < mean < 1,
            "Beta: mean must be strictly between 0 and 1 (it's a probability/proportion).",
        )
        max_var = mean * (1 - mean)
        _require(
            variance is not None and 0 < variance < max_var,
            f"Beta: variance must be between 0 and mean*(1-mean) = {max_var:.6g} for "
            f"mean={mean:.6g} (a Beta can't be more spread out than that and stay in [0, 1]).",
        )
        common = mean * (1 - mean) / variance - 1
        alpha = mean * common
        beta = (1 - mean) * common
        params = {"a": float(alpha), "b": float(beta)}
        dist = stats.beta(**params)

    elif family is Family.POISSON:
        _require(mean is not None and mean > 0, "Poisson: mean must be > 0.")
        if variance is not None and not np.isclose(variance, mean, rtol=1e-6):
            notes.append(
                "Poisson forces variance == mean (it has only one free parameter); the "
                f"requested variance ({variance:.6g}) was ignored and variance={mean:.6g} was "
                "used instead. Use Negative Binomial if you need mean and variance set "
                "independently (e.g. over-dispersed task/lead counts)."
            )
            warnings.warn(notes[-1], stacklevel=2)
        params = {"mu": float(mean)}
        dist = stats.poisson(**params)

    elif family is Family.LOGNORMAL:
        _require(mean is not None and mean > 0, "Lognormal: mean must be > 0.")
        _require(variance is not None and variance > 0, "Lognormal: variance must be > 0.")
        sigma2 = np.log1p(variance / mean**2)
        mu = np.log(mean) - sigma2 / 2
        params = {"s": float(np.sqrt(sigma2)), "scale": float(np.exp(mu))}
        dist = stats.lognorm(**params)

    elif family is Family.NEGATIVE_BINOMIAL:
        _require(mean is not None and mean > 0, "Negative Binomial: mean must be > 0.")
        _require(
            variance is not None and variance > mean,
            "Negative Binomial: variance must be strictly greater than mean (that's what makes "
            "it 'over-dispersed' relative to Poisson; if variance == mean, use Poisson instead)."
        )
        p = mean / variance
        n = mean**2 / (variance - mean)
        params = {"n": float(n), "p": float(p)}
        dist = stats.nbinom(**params)

    else:  # pragma: no cover - Family enum exhausts every branch above
        raise DistributionConfigError(f"Unsupported family: {family!r}")

    return DistributionResult(
        family=family,
        mean=float(mean),
        variance=None if variance is None else float(variance),
        native_params=params,
        dist=dist,
        warnings=tuple(notes),
    )


def uniform_from_bounds(family: Family, low: float | None, high: float | None) -> DistributionResult:
    """Build a Discrete- or Continuous-Uniform distribution directly from its two endpoints,
    rather than via method-of-moments: a Uniform is already fully determined by (low, high), so
    asking for "mean and variance" instead would just make the user do that low/high -> mean/
    variance conversion in their head (Phase 1 feedback: "the two parameters should be the left
    and right endpoints").

    Mirrors `moments_to_params`'s contract: returns the same `DistributionResult` shape (so
    `preview_xy`/`sample_positive`/etc. all work unmodified), and raises DistributionConfigError
    -- never silently clamps -- for an out-of-domain (low, high).
    """
    family = Family(family)
    _require(
        low is not None and high is not None,
        f"{family.value}: both a low and a high endpoint are required.",
    )
    _require(high > low, f"{family.value}: high ({high!r}) must be greater than low ({low!r}).")

    if family is Family.DISCRETE_UNIFORM:
        lo_i, hi_i = int(round(low)), int(round(high))
        _require(
            hi_i > lo_i,
            f"{family.value}: high and low must differ by at least 1 once rounded to whole numbers.",
        )
        dist = stats.randint(lo_i, hi_i + 1)  # scipy randint's high is exclusive -- +1 keeps hi_i.
        native_params = {"low": float(lo_i), "high": float(hi_i)}
    elif family is Family.CONTINUOUS_UNIFORM:
        dist = stats.uniform(loc=float(low), scale=float(high - low))
        native_params = {"low": float(low), "high": float(high)}
    else:  # pragma: no cover - only called for the two Uniform families
        raise DistributionConfigError(f"{family!r} is not a uniform family.")

    return DistributionResult(
        family=family,
        mean=float(dist.mean()),
        variance=float(dist.var()),
        native_params=native_params,
        dist=dist,
        warnings=(),
    )


def preview_xy(
    result: DistributionResult, n_points: int = 200, lower_bound: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Build (x, y) arrays for the distribution-picker preview plot (PLAN.md section 7).

    Continuous families return a pdf curve; discrete families (Poisson, Negative Binomial)
    return a pmf evaluated at each integer in range, which the UI should render as a bar/stem
    plot rather than a line.

    `lower_bound`, when given, previews the *truncated-and-renormalized* density on
    [lower_bound, hi) instead -- the same density `sample_positive` draws from when it rejects
    and redraws values at or below that bound. For a family whose support is already entirely
    above `lower_bound` (Gamma, Lognormal with lower_bound=0) this is identical to the
    untruncated preview; it only actually reshapes the curve for a family like Normal whose
    support extends below the bound.
    """
    dist = result.dist
    if result.family in DISCRETE_FAMILIES:
        hi = int(dist.ppf(0.999))
        hi = max(hi, int(np.ceil(dist.mean())) + 1)
        x = np.arange(0, hi + 1)
        y = dist.pmf(x)
        return x, y

    if lower_bound is None:
        lo, hi = dist.ppf(0.001), dist.ppf(0.999)
        x = np.linspace(lo, hi, n_points)
        y = dist.pdf(x)
        return x, y

    survival = dist.sf(lower_bound)  # Pr(X > lower_bound) -- the truncated density's normalizer.
    _require(
        survival > 1e-9,
        f"{result.family.value}: virtually none of the distribution (mean={result.mean:.6g}) is "
        f"above {lower_bound:.6g}, so there's nothing meaningful to preview.",
    )
    # hi such that the truncated CDF reaches 0.999: CDF(hi) = CDF(lower_bound) + 0.999*survival.
    hi = dist.ppf(1 - 0.001 * survival)
    x = np.linspace(lower_bound, hi, n_points)
    y = dist.pdf(x) / survival
    return x, y


def sample_positive(
    result: DistributionResult,
    size: int,
    rng: np.random.Generator,
    lower_bound: float = 0.0,
    max_rounds: int = 100,
) -> np.ndarray:
    """Draw `size` values from `result.dist`, enforcing `x > lower_bound` by rejection sampling:
    redraw (rather than clamp or discard-and-shrink) until `size` accepted values have been
    collected. The accepted values are therefore genuinely distributed as the analytically
    truncated distribution that `preview_xy(result, lower_bound=...)` previews -- not a clipped
    or shifted approximation of it.

    Converges quickly as long as a comfortable majority of the distribution's mass is already
    above `lower_bound` (for deal size using Normal, PLAN's requirement that mean >= $0 and
    therefore Pr(X > $0) > 0.5 is exactly what keeps this from stalling). Raises
    DistributionConfigError if `max_rounds` batches still aren't enough -- a config that
    shouldn't be reachable through `SimulationConfig.validate()`, but this stays safe on its own
    rather than looping forever.
    """
    accepted = np.empty(0)
    for _ in range(max_rounds):
        draw = np.atleast_1d(result.dist.rvs(size=size, random_state=rng))
        accepted = np.concatenate([accepted, draw[draw > lower_bound]])
        if len(accepted) >= size:
            return accepted[:size]
    raise DistributionConfigError(
        f"Could not draw {size} values above {lower_bound:.6g} for {result.family.value} "
        f"(mean={result.mean:.6g}) after {max_rounds} rounds of resampling -- too little of the "
        "distribution's mass is above the bound."
    )
