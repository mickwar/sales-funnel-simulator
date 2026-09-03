"""Tests for the logit-space effects composition system (PLAN.md sections 1, 4, 6)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit, logit

from funnel_sim.simulation.effects import Effect, EffectSet, apply_effects, to_logit, to_probability


def test_to_logit_and_to_probability_are_inverses():
    probs = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    assert to_probability(to_logit(probs)) == pytest.approx(probs, abs=1e-9)


def test_apply_effects_with_no_deltas_is_identity():
    assert apply_effects(0.3) == pytest.approx(0.3, abs=1e-9)
    assert apply_effects(0.3, []) == pytest.approx(0.3, abs=1e-9)


def test_apply_effects_matches_manual_logit_arithmetic():
    base_prob = 0.2
    deltas = [0.4, -0.15, 0.6]
    expected = expit(logit(base_prob) + sum(deltas))
    assert apply_effects(base_prob, deltas) == pytest.approx(expected)


def test_apply_effects_is_vectorized_over_rows():
    base_prob = np.array([0.1, 0.2, 0.3])
    rep_skill = np.array([0.5, -0.5, 0.0])
    industry = np.array([-0.2, -0.2, -0.2])
    result = apply_effects(base_prob, [rep_skill, industry])

    expected = np.array(
        [
            expit(logit(0.1) + 0.5 - 0.2),
            expit(logit(0.2) - 0.5 - 0.2),
            expit(logit(0.3) + 0.0 - 0.2),
        ]
    )
    assert result == pytest.approx(expected)


def test_extreme_stacked_effects_stay_within_bounds_and_finite():
    result = apply_effects(0.5, [50.0, 50.0, 50.0])
    assert np.isfinite(result)
    assert 0.0 < result <= 1.0

    result = apply_effects(0.5, [-50.0, -50.0, -50.0])
    assert np.isfinite(result)
    assert 0.0 <= result < 1.0


def test_effect_set_total_delta_and_true_effects():
    effects = (
        Effect(name="rep_skill", logit_delta=0.4, description="Rep A is above average"),
        Effect(name="industry_construction", logit_delta=-0.3),
        Effect(name="fast_first_touch", logit_delta=0.6),
    )
    effect_set = EffectSet(base_prob=0.2, effects=effects)

    assert effect_set.total_delta() == pytest.approx(0.4 - 0.3 + 0.6)
    assert effect_set.true_effects() == {
        "rep_skill": 0.4,
        "industry_construction": -0.3,
        "fast_first_touch": 0.6,
    }
    assert effect_set.apply() == pytest.approx(expit(logit(0.2) + 0.4 - 0.3 + 0.6))


def test_effect_set_with_no_effects_returns_base_prob():
    effect_set = EffectSet(base_prob=0.35, effects=())
    assert effect_set.total_delta() == 0.0
    assert effect_set.apply() == pytest.approx(0.35, abs=1e-9)
    assert effect_set.true_effects() == {}


def test_effect_set_supports_per_row_array_deltas():
    base_prob = np.full(4, 0.25)
    rep_skill_per_row = np.array([0.5, 0.5, -0.5, -0.5])
    effects = (Effect(name="rep_skill", logit_delta=rep_skill_per_row),)
    effect_set = EffectSet(base_prob=base_prob, effects=effects)

    expected = expit(logit(base_prob) + rep_skill_per_row)
    assert effect_set.apply() == pytest.approx(expected)
    assert effect_set.true_effects()["rep_skill"] is rep_skill_per_row
