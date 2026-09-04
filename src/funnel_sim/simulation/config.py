"""Statistical/probabilistic parameter configuration for data generation (PLAN.md sections 4
and 7 — the distribution-picker widget's backing model).

This is where "you control the generative process, so you know the ground truth" (PLAN.md
section 1) becomes a concrete, settable object: `SimulationConfig` bundles every tunable
parameter — lead arrival rate, deal size, industry mix, and the logit-space effects that modify
close probability — into one value the fast-forward generator (`generation.py`) samples from,
and that a later fitted model's estimates get compared against.

Built directly on Phase 0's `distributions` and `effects` modules rather than duplicating their
logic: `ParamSpec` wraps a (family, mean, variance) config and resolves it via
`distributions.moments_to_params`; effects stay plain `effects.Effect` instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

from .distributions import DistributionResult, Family, moments_to_params
from .effects import Effect
from .entities import Industry, TaskType


class SimulationConfigError(ValueError):
    """Raised when a SimulationConfig is internally inconsistent (e.g. an industry mix that
    doesn't sum to 1), as opposed to DistributionConfigError, which flags a single bad
    (family, mean, variance) triple. Kept separate so callers can catch either or both.
    """


@dataclass(frozen=True)
class ParamSpec:
    """A user-facing statistical parameter: pick a family, give a mean and (usually) a
    variance. Thin wrapper around `distributions.moments_to_params` — `resolve()` is where the
    method-of-moments conversion and domain validation actually happen, so a `ParamSpec` can be
    stored/passed around cheaply (e.g. in Streamlit's `session_state`) without eagerly building
    a scipy distribution object.
    """

    family: Family
    mean: float
    variance: float | None = None

    def resolve(self) -> DistributionResult:
        """Convert to a usable scipy distribution. Raises DistributionConfigError if
        (family, mean, variance) is outside that family's valid domain — see distributions.py.
        """
        return moments_to_params(self.family, self.mean, self.variance)


# Families appropriate for a daily arrival-count parameter (PLAN.md section 4: prefer Negative
# Binomial over Poisson when the user wants mean and variance set independently).
COUNT_FAMILIES = (Family.POISSON, Family.NEGATIVE_BINOMIAL)

# Families offered for deal size. Normal is included (it's the most intuitive to most people)
# even though its support isn't naturally positive -- `validate_deal_size_spec` below requires a
# non-negative mean, and actual sampling enforces the $0 floor by rejection (see
# distributions.sample_positive), never by clamping or shifting the distribution.
DEAL_SIZE_FAMILIES = (Family.NORMAL, Family.GAMMA, Family.LOGNORMAL)


def validate_deal_size_spec(spec: ParamSpec) -> None:
    """Deal-size-specific validation beyond the generic per-family domain checks
    `ParamSpec.resolve()` already does. Only Normal needs an extra rule here: Gamma and
    Lognormal are positive-supported by construction, but Normal's support extends below $0, so
    a mean that's already negative would mean *most* of the distribution needs to be rejected
    and redrawn (or, worse, none of it converges) -- requiring mean >= $0 guarantees
    Pr(X > $0) > 0.5, which is what keeps `distributions.sample_positive`'s rejection loop
    converging quickly.
    """
    if spec.family is Family.NORMAL and spec.mean < 0:
        raise SimulationConfigError(
            f"Deal size using Normal needs a mean of $0 or more (got ${spec.mean:,.2f}) so that "
            "more than half the distribution is already above $0 -- Pr(X > $0) > 0.5."
        )


def _default_industry_mix() -> dict[Industry, float]:
    # Equal weight across every configured industry — a deliberately neutral starting point;
    # the UI lets a user skew this (PLAN.md section 12: "seeded realistic defaults" is an open
    # question, punted for now in favor of "fully user-configurable from a blank slate").
    n = len(Industry)
    return {industry: 1.0 / n for industry in Industry}


def _default_activity_type_mix() -> dict[TaskType, float]:
    # Equal weight across every TaskType (call/email/meeting/text) — same "deliberately neutral
    # starting point, user-configurable" rationale as _default_industry_mix above.
    n = len(TaskType)
    return {task_type: 1.0 / n for task_type in TaskType}


@dataclass
class SimulationConfig:
    """Every statistical parameter that drives one run's data generation.

    Not frozen: the Streamlit app mutates a working copy via `dataclasses.replace` as the user
    adjusts sliders/pickers (see `with_lead_arrival`, etc.), then calls `validate()` before
    handing it to the generator.
    """

    seed: int
    lead_arrival: ParamSpec  # leads/day — Poisson or Negative Binomial (see COUNT_FAMILIES).
    deal_size: ParamSpec  # dollars — Normal, Gamma, or Lognormal (see DEAL_SIZE_FAMILIES).
    base_close_prob: float = 0.2  # baseline win probability before any effects are applied.
    industry_mix: Mapping[Industry, float] = field(default_factory=_default_industry_mix)
    # True configured effects, in logit space (PLAN.md sections 1/4/6) — what a later fitted
    # model's recovered coefficients should be compared against. Keyed by industry / rep_id so
    # the generator can look a row's effect up by its assigned industry or rep.
    industry_effects: Mapping[Industry, Effect] = field(default_factory=dict)
    rep_skill_effects: Mapping[str, Effect] = field(default_factory=dict)
    # -- Beyond lead creation (PLAN.md section 4's "assign tasks ... sample task outcomes"): the
    # first increment of simulating rep activity on already-created leads and converting some of
    # them into Opportunities. See generation.simulate_activities for how these combine.
    activity_prob: float = 0.3  # chance an open (not yet converted) lead gets a rep activity on
    # any given simulated day. 0 means leads are created but never worked; 1 means every open
    # lead gets touched every day.
    activity_type_mix: Mapping[TaskType, float] = field(default_factory=_default_activity_type_mix)
    lead_conversion_base_prob: float = 0.05  # baseline chance that *one* activity converts its
    # lead into an Opportunity, before any task_type_effects shift -- deliberately per-activity,
    # not per-day, so a lead that never gets touched has exactly 0% chance to convert (PLAN.md
    # section 1: "activities should affect the conversion probability").
    # The *true* configured per-task-type effect on lead_conversion_base_prob, in logit space —
    # same "ground truth a later fitted model should recover" pattern as industry_effects/
    # rep_skill_effects above, and, like them, empty by default (no task type is favored until a
    # scenario explicitly configures one) rather than not yet exposed in the picker UI.
    task_type_effects: Mapping[TaskType, Effect] = field(default_factory=dict)

    def validate(self) -> None:
        """Raise SimulationConfigError (or DistributionConfigError, from a nested ParamSpec) if
        this config can't actually be sampled from. Called explicitly rather than in
        `__post_init__` so a partially-edited config (e.g. mid-keystroke in a Streamlit number
        input) doesn't raise before the user has finished typing.
        """
        if self.lead_arrival.family not in COUNT_FAMILIES:
            raise SimulationConfigError(
                f"lead_arrival must use a count distribution ({', '.join(f.value for f in COUNT_FAMILIES)}), "
                f"got {self.lead_arrival.family.value}."
            )
        if self.deal_size.family not in DEAL_SIZE_FAMILIES:
            raise SimulationConfigError(
                "deal_size must use one of "
                f"({', '.join(f.value for f in DEAL_SIZE_FAMILIES)}), "
                f"got {self.deal_size.family.value}."
            )
        validate_deal_size_spec(self.deal_size)
        if not 0.0 < self.base_close_prob < 1.0:
            raise SimulationConfigError("base_close_prob must be strictly between 0 and 1.")

        mix_total = sum(self.industry_mix.values())
        if not self.industry_mix:
            raise SimulationConfigError("industry_mix must not be empty.")
        if any(weight < 0 for weight in self.industry_mix.values()):
            raise SimulationConfigError("industry_mix weights must all be >= 0.")
        if abs(mix_total - 1.0) > 1e-6:
            raise SimulationConfigError(f"industry_mix weights must sum to 1.0, got {mix_total:.6g}.")

        if not 0.0 <= self.activity_prob <= 1.0:
            raise SimulationConfigError("activity_prob must be between 0 and 1.")
        if not 0.0 < self.lead_conversion_base_prob < 1.0:
            raise SimulationConfigError("lead_conversion_base_prob must be strictly between 0 and 1.")

        activity_mix_total = sum(self.activity_type_mix.values())
        if not self.activity_type_mix:
            raise SimulationConfigError("activity_type_mix must not be empty.")
        if any(weight < 0 for weight in self.activity_type_mix.values()):
            raise SimulationConfigError("activity_type_mix weights must all be >= 0.")
        if abs(activity_mix_total - 1.0) > 1e-6:
            raise SimulationConfigError(
                f"activity_type_mix weights must sum to 1.0, got {activity_mix_total:.6g}."
            )

        # Resolving each distribution also validates it — surfaces a DistributionConfigError
        # with the same clear, user-facing message the picker widget shows (PLAN.md section 4).
        self.lead_arrival.resolve()
        self.deal_size.resolve()

    def with_lead_arrival(self, family: Family, mean: float, variance: float | None) -> "SimulationConfig":
        """Return a copy with a new lead_arrival ParamSpec. Doesn't validate — call
        `validate()` (or `resolve()` on the new ParamSpec directly) when you need the error.
        """
        return replace(self, lead_arrival=ParamSpec(Family(family), mean, variance))

    def with_deal_size(self, family: Family, mean: float, variance: float | None) -> "SimulationConfig":
        """Return a copy with a new deal_size ParamSpec."""
        return replace(self, deal_size=ParamSpec(Family(family), mean, variance))


def default_config(seed: int = 42) -> SimulationConfig:
    """A reasonable out-of-the-box config so the app/tests can fast-forward without the user
    (or a test) configuring every parameter by hand first.

    deal_size defaults to Normal — of the offered families it's the one most people already
    have an intuition for, so it's the friendliest starting point even though Gamma/Lognormal
    are the "naturally positive" choices. lead_arrival can't default to Normal the same way
    (count data needs a count distribution — see COUNT_FAMILIES), so it defaults to Poisson,
    the simpler of the two count families.
    """
    return SimulationConfig(
        seed=seed,
        lead_arrival=ParamSpec(Family.POISSON, mean=20.0),
        deal_size=ParamSpec(Family.NORMAL, mean=5_000.0, variance=200.0**2),
        base_close_prob=0.2,
    )
