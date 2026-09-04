"""Tests for the core data objects (PLAN.md section 3)."""

from __future__ import annotations

import pytest

from funnel_sim.simulation.entities import (
    Account,
    Industry,
    Lead,
    LeadStage,
    Opportunity,
    OpportunityStage,
    Rep,
    RepQuota,
    RevenueBand,
    Run,
    Task,
    TaskType,
)


def test_account_gets_a_unique_id_by_default():
    a = Account(
        run_id="run-1", industry=Industry.SAAS, employee_count=50, revenue_band=RevenueBand.FROM_1M_TO_10M, icp_fit_score=0.6
    )
    b = Account(
        run_id="run-1", industry=Industry.SAAS, employee_count=50, revenue_band=RevenueBand.FROM_1M_TO_10M, icp_fit_score=0.6
    )
    assert a.account_id != b.account_id
    assert a.account_id  # non-empty


def test_lead_defaults_to_new_stage_and_no_assigned_rep():
    lead = Lead(run_id="run-1", account_id="acct-1", created_at_sim_day=3)
    assert lead.stage is LeadStage.NEW
    assert lead.assigned_rep_id is None
    assert lead.lead_id


def test_task_requires_a_lead_or_an_opportunity():
    # Fine: tied to a lead.
    Task(run_id="run-1", task_type=TaskType.CALL, sim_day=1, lead_id="lead-1")
    # Fine: tied to an opportunity.
    Task(run_id="run-1", task_type=TaskType.EMAIL, sim_day=1, opportunity_id="opp-1")

    with pytest.raises(ValueError, match="lead_id or an opportunity_id"):
        Task(run_id="run-1", task_type=TaskType.CALL, sim_day=1)


def test_opportunity_defaults_to_prospecting_and_unresolved_outcome():
    opp = Opportunity(
        run_id="run-1", lead_id="lead-1", account_id="acct-1", deal_size=5000.0, created_at_sim_day=10
    )
    assert opp.stage is OpportunityStage.PROSPECTING
    assert opp.won is None
    assert opp.closed_at_sim_day is None


def test_rep_and_rep_quota_link_by_id():
    rep = Rep(run_id="run-1", name="Alex", capacity_tasks_per_day=20)
    quota = RepQuota(
        run_id="run-1",
        rep_id=rep.rep_id,
        period_start_sim_day=0,
        period_end_sim_day=30,
        target=50_000.0,
    )
    assert quota.rep_id == rep.rep_id
    assert quota.attained == 0.0


def test_run_carries_its_own_seed_for_reproducibility():
    run = Run(name="Baseline scenario", seed=42)
    assert run.seed == 42
    assert run.status == "active"
    assert run.run_id
