"""Tests for the vectorized fast-forward generator (PLAN.md section 4)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from funnel_sim.simulation.config import ParamSpec, SimulationConfig, default_config
from funnel_sim.simulation.distributions import Family
from funnel_sim.simulation.effects import Effect
from funnel_sim.simulation.entities import Industry, LeadStage, TaskType
from funnel_sim.simulation.generation import fast_forward, generate_day, simulate_activities


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
        "days_until_converted",
        "base_conversion_prob",
        "conversion_decay",
        "conversion_prob",
        "deal_size",
    ]


def test_generate_day_assigns_conversion_and_deal_size_fields_to_every_new_lead():
    # Phase 1 feedback: each Lead gets its own days-until-converted, base conversion
    # probability, decay, and deal size *once*, at creation -- not derived from any later
    # activity.
    config = _config_with_guaranteed_leads()
    rng = np.random.default_rng(0)
    result = generate_day(config, sim_day=0, rng=rng, run_id="run-1")

    for col in (
        "days_until_converted",
        "base_conversion_prob",
        "conversion_decay",
        "conversion_prob",
        "deal_size",
    ):
        assert col in result.leads.columns

    assert (result.leads["days_until_converted"] >= 0).all()
    assert (result.leads["base_conversion_prob"] > 0).all()
    assert (result.leads["base_conversion_prob"] < 1).all()
    # conversion_prob starts out equal to base_conversion_prob -- it only evolves once a day is
    # actually simulated (simulate_activities), which generate_day never calls.
    assert (result.leads["conversion_prob"] == result.leads["base_conversion_prob"]).all()
    assert (result.leads["conversion_decay"] > 0).all()
    assert (result.leads["conversion_decay"] < 1).all()
    assert (result.leads["deal_size"] > 0).all()


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
    # Since `.leads` is now a current-state projection (one row per lead, upserted daily -- PLAN.md
    # section 3), the second call must also be handed the first call's leads via `existing_leads`
    # so the activity/conversion draws in part_b see the same open-leads pool a one-shot 6-day call
    # would -- otherwise the rng stream position (which depends on pool size) would diverge.
    config = _config_with_guaranteed_leads()

    shared_rng = np.random.default_rng(99)
    part_a = fast_forward(config, days=3, start_day=0, rng=shared_rng, run_id="run-x")
    part_b = fast_forward(
        config, days=3, start_day=3, rng=shared_rng, run_id="run-x", existing_leads=part_a.leads
    )

    one_shot = fast_forward(config, days=6, start_day=0, seed=99, run_id="run-x")

    continued_counts = part_b.leads.groupby("created_at_sim_day").size().sort_index()
    one_shot_counts = one_shot.leads.groupby("created_at_sim_day").size().sort_index()
    pd.testing.assert_series_equal(continued_counts, one_shot_counts)


def _open_leads(
    n: int,
    stage: LeadStage = LeadStage.NEW,
    created_at_sim_day: int = 0,
    days_until_converted: int = 5,
    conversion_decay: float = 0.8,
    conversion_prob: float = 0.2,
    deal_size: float = 5_000.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lead_id": [f"lead-{i}" for i in range(n)],
            "account_id": [f"acct-{i}" for i in range(n)],
            "stage": [stage.value] * n,
            "created_at_sim_day": [created_at_sim_day] * n,
            "days_until_converted": [days_until_converted] * n,
            "conversion_decay": [conversion_decay] * n,
            "conversion_prob": [conversion_prob] * n,
            "deal_size": [deal_size] * n,
        }
    )


def test_simulate_activities_with_an_empty_pool_returns_well_shaped_empty_frames():
    config = default_config()
    rng = np.random.default_rng(0)
    result = simulate_activities(config, sim_day=0, rng=rng, run_id="run-1", open_leads=_open_leads(0))

    assert result.tasks.empty
    assert result.opportunities.empty
    assert result.lead_updates.empty
    assert list(result.tasks.columns) == [
        "task_id", "run_id", "task_type", "sim_day", "lead_id", "opportunity_id",
        "actor_rep_id", "outcome",
    ]
    assert list(result.lead_updates.columns) == ["lead_id", "stage", "updated_at_sim_day", "conversion_prob"]


def test_simulate_activities_with_zero_activity_prob_logs_no_tasks():
    # conversion_prob=0.0 isolates this to "no activities logged" -- otherwise leads could still
    # convert on schedule with zero rep activity (see the untouched-leads-can-convert test below,
    # which is the actual point of this Phase 1 feedback).
    config = replace(default_config(), activity_prob=0.0)
    rng = np.random.default_rng(0)
    open_leads = _open_leads(50, conversion_prob=0.0)
    result = simulate_activities(config, sim_day=0, rng=rng, run_id="run-1", open_leads=open_leads)

    assert result.tasks.empty
    assert result.opportunities.empty
    # Every open lead still gets a daily update row -- touched or not, it was still evaluated.
    assert len(result.lead_updates) == 50
    assert (result.lead_updates["stage"] == LeadStage.NEW.value).all()


def test_simulate_activities_untouched_leads_can_still_convert():
    # The core Phase 1 feedback this implements: "the chance of a lead being converted should not
    # depend on the outcome of any particular activity." With zero activity at all, a lead with a
    # high conversion_prob still converts on schedule.
    config = replace(default_config(), activity_prob=0.0)
    rng = np.random.default_rng(0)
    open_leads = _open_leads(500, conversion_prob=0.5)
    result = simulate_activities(config, sim_day=0, rng=rng, run_id="run-1", open_leads=open_leads)

    assert result.tasks.empty
    assert not result.opportunities.empty
    converted = (result.lead_updates["stage"] == LeadStage.CONVERTED.value).sum()
    assert converted == len(result.opportunities) > 0


def test_simulate_activities_with_full_activity_prob_touches_every_lead():
    config = replace(default_config(), activity_prob=1.0)
    rng = np.random.default_rng(0)
    open_leads = _open_leads(50)
    result = simulate_activities(config, sim_day=3, rng=rng, run_id="run-1", open_leads=open_leads)

    assert len(result.tasks) == 50
    assert set(result.tasks["lead_id"]) == set(open_leads["lead_id"])
    assert (result.tasks["sim_day"] == 3).all()
    assert (result.tasks["run_id"] == "run-1").all()
    valid_types = {t.value for t in TaskType}
    assert set(result.tasks["task_type"]).issubset(valid_types)
    assert len(result.lead_updates) == 50


def test_simulate_activities_moves_touched_new_leads_to_contacted_when_not_converted():
    config = replace(default_config(), activity_prob=1.0)
    rng = np.random.default_rng(0)
    # A high days_until_converted keeps every non-converting lead from also being disqualified
    # today, isolating this test to the NEW -> CONTACTED transition.
    open_leads = _open_leads(200, stage=LeadStage.NEW, conversion_prob=0.001, days_until_converted=1000)
    result = simulate_activities(config, sim_day=0, rng=rng, run_id="run-1", open_leads=open_leads)

    non_converted = result.lead_updates.loc[result.lead_updates["stage"] != LeadStage.CONVERTED.value]
    assert not non_converted.empty
    assert (non_converted["stage"] == LeadStage.CONTACTED.value).all()
    assert (result.lead_updates["updated_at_sim_day"] == 0).all()


def test_simulate_activities_decays_probability_for_leads_that_do_not_convert():
    config = replace(default_config(), activity_prob=0.0)
    rng = np.random.default_rng(0)
    open_leads = _open_leads(
        2000, conversion_prob=0.2, conversion_decay=0.5, days_until_converted=1000
    )
    result = simulate_activities(config, sim_day=0, rng=rng, run_id="run-1", open_leads=open_leads)

    still_open = result.lead_updates.loc[result.lead_updates["stage"] == LeadStage.NEW.value]
    assert not still_open.empty
    assert still_open["conversion_prob"].to_numpy() == pytest.approx(0.1)  # 0.2 * 0.5 decay


def test_simulate_activities_disqualifies_leads_past_their_days_until_converted():
    config = replace(default_config(), activity_prob=0.0)
    rng = np.random.default_rng(0)
    # conversion_prob=0.0 guarantees no conversion; days_until_converted=0 with age=0 today means
    # today was already this lead's last chance.
    open_leads = _open_leads(10, conversion_prob=0.0, days_until_converted=0, created_at_sim_day=0)
    result = simulate_activities(config, sim_day=0, rng=rng, run_id="run-1", open_leads=open_leads)

    assert (result.lead_updates["stage"] == LeadStage.DISQUALIFIED.value).all()
    assert result.opportunities.empty


def test_simulate_activities_with_certain_conversion_creates_matching_opportunities():
    # conversion_prob just under 1.0 -- apply_effects' logit-space composition means an input of
    # exactly 1.0 would blow up (logit(1.0) is undefined), so this is as close to "always
    # converts" as the probability space allows.
    config = replace(default_config(), activity_prob=1.0)
    rng = np.random.default_rng(0)
    open_leads = _open_leads(100, conversion_prob=0.999999, deal_size=1_234.0)
    result = simulate_activities(config, sim_day=7, rng=rng, run_id="run-1", open_leads=open_leads)

    assert len(result.opportunities) == len(result.tasks) == 100
    assert (result.lead_updates["stage"] == LeadStage.CONVERTED.value).all()
    assert (result.opportunities["created_at_sim_day"] == 7).all()
    assert (result.opportunities["run_id"] == "run-1").all()
    assert set(result.opportunities["lead_id"]) == set(open_leads["lead_id"])
    # Deal size comes from the lead's own pre-assigned value, never a fresh draw at conversion.
    assert (result.opportunities["deal_size"] == 1_234.0).all()
    assert (result.tasks["outcome"] == "advanced").all()


def test_simulate_activities_task_type_effect_shifts_conversion_rate():
    # A large positive logit_delta on CALL should make CALL activities' leads convert far more
    # often than EMAIL activities' leads, which have no configured effect (0.0 shift) -- the
    # effect nudges the lead's conversion_prob, it doesn't decide the roll on its own.
    config = replace(
        default_config(),
        activity_prob=1.0,
        activity_type_mix={TaskType.CALL: 0.5, TaskType.EMAIL: 0.5},
        task_type_effects={TaskType.CALL: Effect(name="call_boost", logit_delta=6.0)},
    )
    rng = np.random.default_rng(0)
    open_leads = _open_leads(4000, conversion_prob=0.05)
    result = simulate_activities(config, sim_day=0, rng=rng, run_id="run-1", open_leads=open_leads)

    tasks = result.tasks
    call_convert_rate = (tasks.loc[tasks["task_type"] == "call", "outcome"] == "advanced").mean()
    email_convert_rate = (tasks.loc[tasks["task_type"] == "email", "outcome"] == "advanced").mean()
    assert call_convert_rate > email_convert_rate + 0.3


def test_fast_forward_produces_tasks_and_opportunities_when_activity_prob_is_high():
    config = _config_with_guaranteed_leads()
    config = replace(
        config,
        activity_prob=1.0,
        base_conversion_prob=ParamSpec(Family.BETA, mean=0.9, variance=0.001),
    )
    result = fast_forward(config, days=5, seed=3)

    assert not result.tasks.empty
    assert not result.opportunities.empty
    assert set(result.tasks["run_id"]) == {result.run_id}
    assert set(result.opportunities["run_id"]) == {result.run_id}
    # Every opportunity's lead should actually show up as CONVERTED in the current-leads
    # projection -- proof the leads_current upsert and the opportunities log agree.
    converted_lead_ids = set(
        result.leads.loc[result.leads["stage"] == LeadStage.CONVERTED.value, "lead_id"]
    )
    assert set(result.opportunities["lead_id"]).issubset(converted_lead_ids)


def test_fast_forward_opportunity_deal_size_matches_its_leads_assigned_deal_size():
    config = _config_with_guaranteed_leads()
    config = replace(
        config,
        activity_prob=1.0,
        base_conversion_prob=ParamSpec(Family.BETA, mean=0.9, variance=0.001),
    )
    result = fast_forward(config, days=3, seed=3)
    assert not result.opportunities.empty

    merged = result.opportunities.merge(
        result.leads[["lead_id", "deal_size"]], on="lead_id", suffixes=("_opp", "_lead")
    )
    assert len(merged) == len(result.opportunities)
    assert (merged["deal_size_opp"] == merged["deal_size_lead"]).all()


def test_fast_forward_with_zero_activity_prob_creates_leads_but_no_tasks():
    config = _config_with_guaranteed_leads()
    config = replace(config, activity_prob=0.0)
    result = fast_forward(config, days=3, seed=3)

    assert not result.leads.empty
    assert result.tasks.empty
    # Leads still convert/get disqualified on schedule with zero rep activity -- conversion was
    # never gated by activity_prob (Phase 1 feedback) -- but nothing should have become CONTACTED
    # (that transition only happens on a touch, and there were none).
    assert set(result.leads["stage"]).issubset(
        {LeadStage.NEW.value, LeadStage.CONVERTED.value, LeadStage.DISQUALIFIED.value}
    )


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
