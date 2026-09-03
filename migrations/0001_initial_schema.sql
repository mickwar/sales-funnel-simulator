-- Initial schema: the event-sourcing model from PLAN.md section 3.
--
-- Two tiers, not one:
--   1. events            append-only, full history, never updated/deleted (except the
--                        retention job in PLAN.md section 5). The audit trail / "time travel"
--                        source -- filtering this table by sim_time is how you answer "what did
--                        the funnel look like on day 47", not a special feature.
--   2. <entity>_current  upserted as each event lands. Every dashboard/ML read hits these, never
--                        the full event log -- replaying hundreds of thousands of events on
--                        every read is the mistake this two-tier design avoids.
--
-- run_id is the leading column of every index (PLAN.md section 5): the app is multi-tenant from
-- day one even though Phase 1 is solo use, so "several concurrent users" later is a hosting
-- change, not a schema change.
--
-- This migration is applied once against a fresh database (see storage/schema.py) -- it is not
-- yet idempotent (no "IF NOT EXISTS"/upsert-style migration tracking), which is fine for a
-- Phase 1 solo-use MVP; add a proper migration runner before Phase 4 opens this up to others.

CREATE TABLE runs (
    run_id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    seed BIGINT NOT NULL,
    config JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE events (
    event_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES runs (run_id),
    entity_type TEXT NOT NULL,
    entity_id UUID NOT NULL,
    event_type TEXT NOT NULL,
    sim_time INTEGER NOT NULL,  -- simulated day (or finer tick) counter, not a wall-clock time.
    payload JSONB NOT NULL,
    actor_rep_id UUID,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX events_run_id_sim_time_idx ON events (run_id, sim_time);
CREATE INDEX events_run_id_entity_idx ON events (run_id, entity_type, entity_id);

CREATE TABLE accounts_current (
    account_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES runs (run_id),
    name TEXT NOT NULL,
    industry TEXT NOT NULL,
    employee_count INTEGER NOT NULL,
    revenue_band TEXT NOT NULL,
    icp_fit_score DOUBLE PRECISION NOT NULL,
    created_at_sim_day INTEGER NOT NULL
);
CREATE INDEX accounts_current_run_id_idx ON accounts_current (run_id);

CREATE TABLE reps_current (
    rep_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES runs (run_id),
    name TEXT NOT NULL,
    capacity_tasks_per_day INTEGER NOT NULL,
    tenure_days INTEGER NOT NULL DEFAULT 0,
    -- True configured skill effect in logit space (PLAN.md sections 1/4) -- the ground truth a
    -- later fitted model's per-rep coefficient is compared against.
    skill_logit_delta DOUBLE PRECISION NOT NULL DEFAULT 0
);
CREATE INDEX reps_current_run_id_idx ON reps_current (run_id);

CREATE TABLE rep_quotas_current (
    quota_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES runs (run_id),
    rep_id UUID NOT NULL REFERENCES reps_current (rep_id),
    period_start_sim_day INTEGER NOT NULL,
    period_end_sim_day INTEGER NOT NULL,
    target DOUBLE PRECISION NOT NULL,
    attained DOUBLE PRECISION NOT NULL DEFAULT 0
);
CREATE INDEX rep_quotas_current_run_id_idx ON rep_quotas_current (run_id);

CREATE TABLE leads_current (
    lead_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES runs (run_id),
    account_id UUID NOT NULL REFERENCES accounts_current (account_id),
    stage TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'unknown',
    assigned_rep_id UUID REFERENCES reps_current (rep_id),
    created_at_sim_day INTEGER NOT NULL,
    updated_at_sim_day INTEGER NOT NULL
);
CREATE INDEX leads_current_run_id_idx ON leads_current (run_id);

CREATE TABLE opportunities_current (
    opportunity_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES runs (run_id),
    lead_id UUID NOT NULL REFERENCES leads_current (lead_id),
    account_id UUID NOT NULL REFERENCES accounts_current (account_id),
    stage TEXT NOT NULL,
    probability DOUBLE PRECISION NOT NULL,
    deal_size DOUBLE PRECISION NOT NULL,
    expected_close_sim_day INTEGER,
    assigned_rep_id UUID REFERENCES reps_current (rep_id),
    created_at_sim_day INTEGER NOT NULL,
    closed_at_sim_day INTEGER,
    won BOOLEAN
);
CREATE INDEX opportunities_current_run_id_idx ON opportunities_current (run_id);

CREATE TABLE tasks_current (
    task_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES runs (run_id),
    lead_id UUID REFERENCES leads_current (lead_id),
    opportunity_id UUID REFERENCES opportunities_current (opportunity_id),
    task_type TEXT NOT NULL,
    actor_rep_id UUID REFERENCES reps_current (rep_id),
    sim_day INTEGER NOT NULL,
    outcome TEXT NOT NULL DEFAULT 'pending',
    CONSTRAINT tasks_current_lead_or_opportunity CHECK (
        lead_id IS NOT NULL OR opportunity_id IS NOT NULL
    )
);
CREATE INDEX tasks_current_run_id_idx ON tasks_current (run_id);
