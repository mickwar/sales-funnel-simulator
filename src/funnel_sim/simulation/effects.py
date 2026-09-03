"""Effects system: compose stacked statistical effects in logit (log-odds) space.

PLAN.md section 4 is the rationale in full; the short version: if "rep A: +15% close rate",
"industry B: -10% close rate", and "fast first touch: +20% close rate" all apply to the same
opportunity, adding or multiplying those percentages directly in probability space breaks down —
stack enough effects and you exceed 100% or go negative. Instead every effect is defined as an
additive shift on the log-odds (logit) scale; shifts are summed, then converted back to a
probability with the sigmoid function. That is exactly the functional form of a logistic
regression, so a model later *fit* to the simulated outcomes is correctly specified, and its
recovered coefficients should land close to the true effects configured here — see PLAN.md
sections 1 and 6 for why that recovery comparison is the app's core demo.

Everything here is vectorized: `base_prob` and each effect's delta can be a scalar or a numpy
array (one entry per simulated row), so a whole fast-forward's worth of outcomes can be generated
without a Python-level loop (PLAN.md section 4's "vectorize generation").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
from scipy.special import expit, logit

# Clip probabilities away from the exact 0/1 boundary before taking a logit. At the float64
# boundary logit(0) / logit(1) are -inf/+inf; propagating an inf through "+ more effects" would
# still resolve correctly via sigmoid, but clipping keeps every intermediate value finite and
# avoids nan from inf - inf style edge cases if effects are ever combined some other way later.
_EPS = 1e-12


def to_logit(prob: float | np.ndarray) -> float | np.ndarray:
    """Convert a probability (or array of probabilities) to log-odds space."""
    clipped = np.clip(prob, _EPS, 1 - _EPS)
    return logit(clipped)


def to_probability(logit_value: float | np.ndarray) -> float | np.ndarray:
    """Convert log-odds (or an array of them) back to a probability via the sigmoid function."""
    return expit(logit_value)


@dataclass(frozen=True)
class Effect:
    """One named, additive shift in log-odds space.

    `logit_delta` is the *true* configured effect — the number a later ground-truth-recovery
    comparison checks a fitted model's coefficient against (PLAN.md section 1). It can be a
    scalar (the same shift for every row) or a numpy array (a per-row shift, e.g. each
    opportunity's assigned rep's skill effect looked up by rep_id).
    """

    name: str
    logit_delta: float | np.ndarray
    description: str = ""


def apply_effects(
    base_prob: float | np.ndarray,
    deltas: Iterable[float | np.ndarray] = (),
) -> float | np.ndarray:
    """Apply a set of logit-space deltas to a base probability and return the resulting
    probability. This is the composition primitive: `base_prob` starts in probability space
    (easy for a human to configure — "the base close rate is 20%"), gets converted to logit
    space, has every delta added, and is converted back.

    With no deltas this is the identity function (up to the float64 round-trip through logit/
    sigmoid), so a row with no configured effects keeps its base probability.
    """
    total = to_logit(base_prob)
    for delta in deltas:
        total = total + delta
    return to_probability(total)


@dataclass(frozen=True)
class EffectSet:
    """A named collection of Effects to apply together to one base probability.

    Bundles the effects so their true configured values travel together with the composed
    probability — exactly the "ground truth" side of the recovery demo in PLAN.md sections 1/6:
    once outcomes are sampled from `.apply(base_prob)`, `.true_effects()` is what a fitted
    model's estimated coefficients get compared against.
    """

    base_prob: float | np.ndarray
    effects: tuple[Effect, ...] = ()

    def total_delta(self) -> float | np.ndarray:
        """Sum of every effect's logit_delta (0.0 if there are no effects)."""
        total: float | np.ndarray = 0.0
        for effect in self.effects:
            total = total + effect.logit_delta
        return total

    def apply(self) -> float | np.ndarray:
        """The resulting probability after applying every effect to `base_prob`."""
        return apply_effects(self.base_prob, (e.logit_delta for e in self.effects))

    def true_effects(self) -> Mapping[str, float | np.ndarray]:
        """The configured ground truth, keyed by effect name — what a fitted model's estimated
        effects should be compared against later (PLAN.md section 1).
        """
        return {effect.name: effect.logit_delta for effect in self.effects}
