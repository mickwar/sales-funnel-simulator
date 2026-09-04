"""Tests for the vectorized fast-forward generator (PLAN.md section 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from funnel_sim.simulation.config import ParamSpec, SimulationConfig, default_config
from funnel_sim.simulation.distributions import Family
from funnel_sim.simulation.entities import Industry
from funnel_sim.simulation.generation import fast_forward, generate_day


def _config_with_guaranteed_leads() -> SimulationConfig:
    # A high mean makes a zero-lead day astronomically unlikely, so tests that need at least
    # one lead don't have to special-case it.
    config = default_config()
    return config.with_lead_arrival(Family.POISSON, mean=50.0, variance=None)


def test_generate_day_produces_matching_accounts_and_leads():
    config = _config_with_guaranteed_leads()
    rng = np.random.default_rng(0)
    result = generate_day(config, sim_day=5, rng=rng, run_id="run-abc")

    assert len(result.accounts) == len(result.leads) > 0
    assert set(result.accounts["account_id"]) == set(result.leads["account_id"])
    assert (result.accounts["run_id"] == "run-abc").all()
    assert (result.leads["run_id"] == "run-abc").all()
    assert (result.accounts["created_at_sim_day"] == 5).all()
    assert (result.leads["created_at_sim_day"] == 5).all()
    assert (result.leads["stage"] == "new").all()

    # Every sampled industry is a real Industry value.
    valid_industries = {i.value for i in Industry}
    assert set(result.accounts["industry"]).issubset(valid_industries)


def test_generate_day_with_near_zero_arrival_can_produce_an_empty_but_well_shaped_day():
    config = default_config()
    # Mean so small that a zero-lead day is close to certain with this fixed seed; the point of
    # this test is that the *shape* (columns, empty-not-None) holds even then.
    config = config.with_lead_arrival(Family.POISSON, mean=0.001, variance=None)
    rng = np.random.default_rng(1)
    result = generate_day(config, sim_day=0, rng=rng, run_id="run-empty")

    assert list(result.accounts.columns) == [
        "account_id",
        "run_id",
        "name",
        "industry",
        "employee_count",
        "revenue_band",
        "icp_fit_score",
        "created_at_sim_day",
    ]
    assert list(result.leads.columns) == [
        "lead_id",
        "run_id",
        "account_id",
        "stage",
        "source",
        "assigned_rep_id",
        "created_at_sim_day",
        "updated_at_sim_day",
    ]


def test_fast_forward_concatenates_every_day_and_stamps_sim_day():
    config = _config_with_guaranteed_leads()
    result = fast_forward(config, days=5, start_day=100, seed=7)

    assert result.days == 5
    assert result.start_day == 100
    assert set(result.leads["created_at_sim_day"]) == set(range(100, 105))
    assert len(result.accounts) == len(result.leads)


def test_fast_forward_is_reproducible_given_the_same_seed():
    config = _config_with_guaranteed_leads()
    result_a = fast_forward(config, days=10, seed=123)
    result_b = fast_forward(config, days=10, seed=123)

    pd.testing.assert_frame_equal(result_a.leads.drop(columns=["lead_id", "account_id", "run_id"]),
                                   result_b.leads.drop(columns=["lead_id", "account_id", "run_id"]))
    assert len(result_a.leads) == len(result_b.leads)


def test_fast_forward_with_different_seeds_produces_different_output():
    config = _config_with_guaranteed_leads()
    result_a = fast_forward(config, days=10, seed=1)
    result_b = fast_forward(config, days=10, seed=2)

    assert len(result_a.leads) != len(result_b.leads) or not result_a.leads[
        "industry"
    ].equals(result_b.leads["industry"])


def test_fast_forward_assigns_one_run_id_to_every_generated_row_and_mints_one_if_absent():
    config = _config_with_guaranteed_leads()
    result = fast_forward(config, days=3, seed=9)

    assert result.run_id
    assert (result.accounts["run_id"] == result.run_id).all()
    assert (result.leads["run_id"] == result.run_id).all()


def test_fast_forward_reuses_a_supplied_run_id():
    config = _config_with_guaranteed_leads()
    result = fast_forward(config, days=2, seed=9, run_id="existing-run")
    assert result.run_id == "existing-run"
    assert (result.leads["run_id"] == "existing-run").all()


def test_fast_forward_rejects_fewer_than_one_day():
    config = default_config()
    with pytest.raises(ValueError, match="days must be >= 1"):
        fast_forward(config, days=0)


def test_fast_forward_with_a_shared_rng_appends_instead_of_replaying():
    # Two calls sharing one rng object should draw *different* leads-per-day counts than a fresh
    # seeded-from-scratch call would (Phase 1 feedback: "Fast forward should append to the
    # previously generated runs, not overwrite them") -- contrast with two independently
    # fresh-seeded calls (rng=None both times), which *do* replay identical draws.
    config = _config_with_guaranteed_leads()
    shared_rng = np.random.default_rng(42)

    first = fast_forward(config, days=3, start_day=0, rng=shared_rng, run_id="run-continued")
    second = fast_forward(config, days=3, start_day=3, rng=shared_rng, run_id="run-continued")

    assert set(first.leads["created_at_sim_day"]) == set(range(0, 3))
    assert set(second.leads["created_at_sim_day"]) == set(range(3, 6))

    # Two *fresh*-seeded calls (no shared rng) with the same seed replay the exact same draws --
    # this is the "overwrite" behavior the shared-rng path is meant to avoid.
    replay_a = fast_forward(config, days=3, start_day=0, seed=42, run_id="run-continued")
    replay_b = fast_forward(config, days=3, start_day=0, seed=42, run_id="run-continued")
    pd.testing.assert_series_equal(
        replay_a.leads.groupby("created_at_sim_day").size(),
        replay_b.leads.groupby("created_at_sim_day").size(),
    )
    # The shared-rng continuation's second batch is a genuinely new slice of the stream, not a
    # replay of the first batch's counts.
    first_counts = first.leads.groupby("created_at_sim_day").size().to_numpy()
    second_counts = second.leads.groupby("created_at_sim_day").size().to_numpy()
    assert not np.array_equal(first_counts, second_counts)


def test_fast_forward_with_a_shared_rng_reuses_the_given_run_id_across_calls():
    config = _config_with_guaranteed_leads()
    shared_rng = np.random.default_rng(7)

    first = fast_forward(config, days=2, start_day=0, rng=shared_rng, run_id="run-continued")
    second = fast_forward(config, days=2, start_day=2, rng=shared_rng, run_id=first.run_id)

    assert first.run_id == "run-continued"
    assert second.run_id == first.run_id
    assert (first.leads["run_id"] == "run-continued").all()
    assert (second.leads["run_id"] == "run-continued").all()


def test_fast_forward_two_calls_sharing_an_rng_matches_one_combined_call():
    # Splitting one continuous rng stream across two `fast_forward` calls (3 days, then 3 more)
    # should be indistinguishable from consuming that same stream in a single 6-day call -- proof
    # that "append" really does continue the simulation rather than skipping or repeating draws.
    config = _config_with_guaranteed_leads()

    shared_rng = np.random.default_rng(99)
    part_a = fast_forward(config, days=3, start_day=0, rng=shared_rng, run_id="run-x")
    part_b = fast_forward(config, days=3, start_day=3, rng=shared_rng, run_id="run-x")
    continued_leads_by_day = pd.concat([part_a.leads, part_b.leads], ignore_index=True)

    one_shot = fast_forward(config, days=6, start_day=0, seed=99, run_id="run-x")

    continued_counts = continued_leads_by_day.groupby("created_at_sim_day").size().sort_index()
    one_shot_counts = one_shot.leads.groupby("created_at_sim_day").size().sort_index()
    pd.testing.assert_series_equal(continued_counts, one_shot_counts)


def test_industry_mix_is_respected_at_scale():
    # Skew heavily toward SAAS and check the sampled distribution roughly matches -- a
    # statistical (not exact) check, so keep n large and the tolerance loose to avoid flakiness.
    config = default_config()
    remaining = [i for i in Industry if i is not Industry.SAAS]
    industry_mix = {i: 0.1 / len(remaining) for i in remaining}
    industry_mix[Industry.SAAS] = 0.9
    config.industry_mix = industry_mix
    config = config.with_lead_arrival(Family.POISSON, mean=500.0, variance=None)

    result = fast_forward(config, days=1, seed=42)
    saas_share = (result.leads.merge(result.accounts[["account_id", "industry"]], on="account_id")
                  ["industry"] == Industry.SAAS.value).mean()
    assert saas_share == pytest.approx(0.9, abs=0.05)
