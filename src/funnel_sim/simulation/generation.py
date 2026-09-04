"""Vectorized, time-stepped data generation — the fast-forward engine (PLAN.md section 4 and
section 10 Phase 1).

Per PLAN.md section 4: "fast forward N days" advances one daily tick at a time; within a tick,
every statistical draw is generated as numpy arrays in one shot rather than looped row-by-row in
Python — that's the difference between a fast-forward that takes seconds and one that takes
minutes at the row counts this app targets (hundreds of thousands of rows/session, PLAN.md
section 5). Constructing one entity object per generated row (to get each its own id) is still a
Python-level loop below, but it does no sampling — every random draw happens once, vectorized,
before that loop runs.

Two things happen each simulated day, in order:

1. `generate_day` — new Leads (and their Accounts) arrive, from `SimulationConfig.lead_arrival`
   and `.industry_mix` (§4's Poisson/Negative-Binomial arrival process).
2. `simulate_activities` — reps work the pool of already-open Leads (new ones included, the same
   day they arrive): some get a randomized activity (call/email/meeting/text, from
   `.activity_type_mix`), and a touched lead has a chance (`.lead_conversion_base_prob`, shifted
   by that task type's `.task_type_effects`) of converting into an Opportunity. A lead that never
   gets touched has exactly 0% chance of converting that day — see `simulate_activities`'s
   docstring.

`fast_forward` is what threads these together across many days, maintaining a leads-*current*
projection (PLAN.md section 3: `<entity>_current` — one row per lead, upserted as its stage
changes) alongside the append-only Task/Opportunity logs.

Honestly-stated scope for this increment: Rep capacity limits ("assign tasks up to each rep's
remaining capacity", §4) aren't modeled — there's no Rep pool generated yet, so every Task's
`actor_rep_id` is `None` and `activity_prob` applies uniformly rather than being capacity-gated.
Opportunities are created but never progressed past `PROSPECTING`/never closed — win/loss
resolution, `industry_effects`, and `rep_skill_effects` are still the next increment.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .distributions import sample_positive
from .effects import Effect, apply_effects
from .entities import (
    Account,
    Industry,
    Lead,
    LeadStage,
    Opportunity,
    RevenueBand,
    Task,
    TaskOutcome,
    TaskType,
)

_ACCOUNT_COLUMNS = [
    "account_id",
    "run_id",
    "name",
    "industry",
    "employee_count",
    "revenue_band",
    "icp_fit_score",
    "created_at_sim_day",
]

_LEAD_COLUMNS = [
    "lead_id",
    "run_id",
    "account_id",
    "stage",
    "source",
    "assigned_rep_id",
    "created_at_sim_day",
    "updated_at_sim_day",
]

_TASK_COLUMNS = [
    "task_id",
    "run_id",
    "task_type",
    "sim_day",
    "lead_id",
    "opportunity_id",
    "actor_rep_id",
    "outcome",
]

_OPPORTUNITY_COLUMNS = [
    "opportunity_id",
    "run_id",
    "lead_id",
    "account_id",
    "deal_size",
    "created_at_sim_day",
    "stage",
    "probability",
    "expected_close_sim_day",
    "assigned_rep_id",
    "closed_at_sim_day",
    "won",
]

_LEAD_UPDATE_COLUMNS = ["lead_id", "stage", "updated_at_sim_day"]

# Leads in any of these stages are still "in play" -- eligible for a rep activity on a given
# simulated day. DISQUALIFIED/CONVERTED are terminal; nothing generates DISQUALIFIED yet (that's
# future work), but the check is written against the general "not terminal" rule rather than an
# explicit allow-list of {NEW, CONTACTED} so it doesn't silently stop including QUALIFIED leads
# if/when something starts producing that stage.
_TERMINAL_LEAD_STAGES = frozenset({LeadStage.CONVERTED.value, LeadStage.DISQUALIFIED.value})

# Employee count and ICP-fit are sampled from fixed, coarse distributions for Phase 1 — not yet
# user-configurable. Revisit alongside PLAN.md section 12's "seeded realistic defaults" question
# if these need to vary by industry or become tunable in their own right.
_EMPLOYEE_COUNT_MEAN = 150.0
_EMPLOYEE_COUNT_SIGMA = 1.2  # lognormal shape parameter (not variance) — controls right-skew.
_ICP_FIT_ALPHA_BETA = (2.0, 2.0)  # Beta(2, 2): symmetric, mass concentrated around 0.5.

_REVENUE_BANDS = tuple(RevenueBand)
# Roughly correlated with the employee-count distribution above — small companies far more
# common than large ones. A crude fixed weighting for Phase 1; revisit if the app needs revenue
# band to actually respond to the sampled employee count rather than being drawn independently.
_REVENUE_BAND_WEIGHTS = (0.5, 0.3, 0.15, 0.05)

# Of the activities that *don't* convert their lead, the share that at least connects with the
# person (TaskOutcome.CONNECTED) rather than going unanswered (NO_RESPONSE) -- a fixed internal
# constant for this increment, like _EMPLOYEE_COUNT_MEAN above, not yet its own config knob.
_TASK_OUTCOME_CONNECT_SHARE = 0.5

# Effect.logit_delta looked up for a task type with no entry in config.task_type_effects -- 0.0
# shift, i.e. "no configured effect yet" (SimulationConfig.task_type_effects's default).
_ZERO_EFFECT = Effect(name="none", logit_delta=0.0)


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(dtype="object") for col in columns})


@dataclass(frozen=True)
class DayResult:
    """One simulated day's worth of newly-*arrived* entities, as DataFrames (one row per entity)
    with columns matching `entities.Account`/`entities.Lead` field names 1:1 — that
    correspondence is what makes bulk-loading these into Postgres (PLAN.md section 5) a direct
    column mapping rather than a translation layer.

    Empty (zero-lead) days still return DataFrames with the right columns, never `None` — so
    callers can always `pd.concat` a sequence of `DayResult`s without special-casing.
    """

    sim_day: int
    accounts: pd.DataFrame
    leads: pd.DataFrame


@dataclass(frozen=True)
class ActivityResult:
    """One simulated day's worth of rep activity on an already-open pool of leads: every
    activity performed as a `Task` row (whether or not it converted anything -- an append-only
    event log, PLAN.md section 3), any `Opportunity` rows created by a converting activity, and
    `lead_updates` -- one row per *touched* lead with its post-activity stage and
    `updated_at_sim_day`, for the caller to upsert into its leads-current projection (touched but
    non-converting leads still get an updated `updated_at_sim_day`, since "last worked" is
    meaningful even when the stage itself doesn't change).

    Empty (nothing touched) days still return DataFrames with the right columns, never `None` —
    same rationale as `DayResult`.
    """

    sim_day: int
    tasks: pd.DataFrame
    opportunities: pd.DataFrame
    lead_updates: pd.DataFrame


@dataclass(frozen=True)
class FastForwardResult:
    """The output of fast-forwarding a SimulationConfig for a number of days.

    `run_id` is the id every generated row carries (PLAN.md section 5's multi-tenancy column) —
    a fresh one is minted per call unless the caller passes one in (e.g. to keep generating into
    an existing Run/scenario).

    `accounts`, `tasks`, and `opportunities` are append-only logs (every row ever generated,
    across every day of this call); `leads` is a *current-state projection* instead — one row per
    lead, reflecting its latest stage/`updated_at_sim_day` after every day's activity, not a
    per-day history (PLAN.md section 3's `<entity>_current` pattern). Pass this call's `.leads`
    back in as `fast_forward`'s `existing_leads` to continue the same projection into a later
    call rather than losing track of which leads are already open/converted.
    """

    days: int
    start_day: int
    run_id: str
    accounts: pd.DataFrame
    leads: pd.DataFrame
    tasks: pd.DataFrame
    opportunities: pd.DataFrame


def generate_day(
    config: SimulationConfig, sim_day: int, rng: np.random.Generator, run_id: str
) -> DayResult:
    """Generate one day's new Accounts and Leads (one Account per new Lead, for now — see
    module docstring) by sampling `config.lead_arrival` for a count, then drawing that many
    industries from `config.industry_mix` and employee counts / ICP-fit scores from fixed
    distributions, all in one vectorized pass (no per-lead sampling loop).
    """
    arrival_dist = config.lead_arrival.resolve().dist
    n_new = max(int(arrival_dist.rvs(random_state=rng)), 0)

    if n_new == 0:
        return DayResult(sim_day, _empty_frame(_ACCOUNT_COLUMNS), _empty_frame(_LEAD_COLUMNS))

    industries = list(config.industry_mix.keys())
    weights = np.array(list(config.industry_mix.values()), dtype=float)
    weights = weights / weights.sum()  # defensive re-normalize; validate() already checks this.
    sampled_industries = rng.choice(np.array([i.value for i in industries]), size=n_new, p=weights)

    mu = np.log(_EMPLOYEE_COUNT_MEAN) - (_EMPLOYEE_COUNT_SIGMA**2) / 2
    employee_counts = rng.lognormal(mean=mu, sigma=_EMPLOYEE_COUNT_SIGMA, size=n_new)
    employee_counts = np.maximum(1, np.round(employee_counts)).astype(int)

    icp_fit_scores = rng.beta(*_ICP_FIT_ALPHA_BETA, size=n_new)
    revenue_bands = rng.choice(
        np.array([b.value for b in _REVENUE_BANDS]), size=n_new, p=_REVENUE_BAND_WEIGHTS
    )

    # One Account + one Lead per new-lead draw. Building the entity objects (for their ids) is
    # a Python-level loop, but every value going into it was already sampled above in bulk —
    # this loop does no randomness of its own.
    accounts: list[Account] = []
    leads: list[Lead] = []
    for i in range(n_new):
        account = Account(
            run_id=run_id,
            industry=Industry(sampled_industries[i]),
            employee_count=int(employee_counts[i]),
            revenue_band=RevenueBand(revenue_bands[i]),
            icp_fit_score=float(icp_fit_scores[i]),
            name=f"Account {sim_day}-{i}",
            created_at_sim_day=sim_day,
        )
        accounts.append(account)
        leads.append(
            Lead(
                run_id=account.run_id,
                account_id=account.account_id,
                created_at_sim_day=sim_day,
                stage=LeadStage.NEW,
                source="generated",
            )
        )

    accounts_df = pd.DataFrame(
        {
            "account_id": [a.account_id for a in accounts],
            "run_id": [a.run_id for a in accounts],
            "name": [a.name for a in accounts],
            "industry": [a.industry.value for a in accounts],
            "employee_count": [a.employee_count for a in accounts],
            "revenue_band": [a.revenue_band.value for a in accounts],
            "icp_fit_score": [a.icp_fit_score for a in accounts],
            "created_at_sim_day": [a.created_at_sim_day for a in accounts],
        }
    )
    leads_df = pd.DataFrame(
        {
            "lead_id": [l.lead_id for l in leads],
            "run_id": [l.run_id for l in leads],
            "account_id": [l.account_id for l in leads],
            "stage": [l.stage.value for l in leads],
            "source": [l.source for l in leads],
            "assigned_rep_id": [l.assigned_rep_id for l in leads],
            "created_at_sim_day": [l.created_at_sim_day for l in leads],
            "updated_at_sim_day": [l.updated_at_sim_day for l in leads],
        }
    )

    return DayResult(sim_day=sim_day, accounts=accounts_df, leads=leads_df)


def simulate_activities(
    config: SimulationConfig,
    sim_day: int,
    rng: np.random.Generator,
    run_id: str,
    open_leads: pd.DataFrame,
) -> ActivityResult:
    """Sample one day's rep activity on `open_leads` (any not-yet-terminal lead — see
    `fast_forward`, which is what builds this pool and decides what counts as "open"; expects at
    least `lead_id`, `account_id`, and `stage` columns).

    For each open lead: roll whether it gets touched today (`config.activity_prob`), and if so,
    which `TaskType` the activity is (`config.activity_type_mix`). Only a *touched* lead has any
    chance of converting that day -- "no touch, no progress" is the whole point of modeling
    activities at all, per the Phase 1 feedback this implements ("activities should affect the
    conversion probability"). A touched lead's conversion roll uses `config.
    lead_conversion_base_prob`, shifted in logit space by that activity's task type's configured
    effect (`config.task_type_effects`, 0.0 if that type has none configured yet) via the same
    `effects.apply_effects` composition every other probability in this app goes through.

    Every touched lead that was `NEW` moves to `CONTACTED` (the activity itself is what
    "contacted" means); a lead whose activity converts it moves to `CONVERTED` instead, and gets
    a new `Opportunity` (deal size drawn from `config.deal_size`, floored at $0 the same way the
    Configure Parameters preview promises -- see `distributions.sample_positive`).

    Fully vectorized, like `generate_day`: every random draw happens once across the whole
    `open_leads` pool, never per-row in a loop -- only building the Task/Opportunity objects
    (for their ids) loops in Python, and does no sampling of its own.
    """
    if open_leads.empty:
        return ActivityResult(
            sim_day=sim_day,
            tasks=_empty_frame(_TASK_COLUMNS),
            opportunities=_empty_frame(_OPPORTUNITY_COLUMNS),
            lead_updates=_empty_frame(_LEAD_UPDATE_COLUMNS),
        )

    touched_mask = rng.random(len(open_leads)) < config.activity_prob
    touched = open_leads.loc[touched_mask]
    n_touched = len(touched)
    if n_touched == 0:
        return ActivityResult(
            sim_day=sim_day,
            tasks=_empty_frame(_TASK_COLUMNS),
            opportunities=_empty_frame(_OPPORTUNITY_COLUMNS),
            lead_updates=_empty_frame(_LEAD_UPDATE_COLUMNS),
        )

    task_types = list(config.activity_type_mix.keys())
    type_weights = np.array(list(config.activity_type_mix.values()), dtype=float)
    type_weights = type_weights / type_weights.sum()  # defensive re-normalize, as generate_day.
    sampled_types = rng.choice(np.array([t.value for t in task_types]), size=n_touched, p=type_weights)

    # This activity's chance to convert its lead: the configured baseline, shifted by whichever
    # task type it happens to be (0.0 -- no shift -- for any type without a configured effect).
    type_deltas = np.array(
        [config.task_type_effects.get(TaskType(t), _ZERO_EFFECT).logit_delta for t in sampled_types]
    )
    convert_prob = apply_effects(config.lead_conversion_base_prob, (type_deltas,))
    converts = rng.random(n_touched) < convert_prob

    # Of the activities that don't convert their lead, split outcomes between "connected" and
    # "no response" for believable Task variety (Phase 1 fixed constant -- see
    # _TASK_OUTCOME_CONNECT_SHARE above).
    connects = rng.random(n_touched) < _TASK_OUTCOME_CONNECT_SHARE
    outcomes = np.where(
        converts,
        TaskOutcome.ADVANCED.value,
        np.where(connects, TaskOutcome.CONNECTED.value, TaskOutcome.NO_RESPONSE.value),
    )

    touched_lead_ids = touched["lead_id"].to_numpy()
    tasks: list[Task] = [
        Task(
            run_id=run_id,
            task_type=TaskType(sampled_types[i]),
            sim_day=sim_day,
            lead_id=str(touched_lead_ids[i]),
            outcome=TaskOutcome(outcomes[i]),
        )
        for i in range(n_touched)
    ]
    tasks_df = pd.DataFrame(
        {
            "task_id": [t.task_id for t in tasks],
            "run_id": [t.run_id for t in tasks],
            "task_type": [t.task_type.value for t in tasks],
            "sim_day": [t.sim_day for t in tasks],
            "lead_id": [t.lead_id for t in tasks],
            "opportunity_id": [t.opportunity_id for t in tasks],
            "actor_rep_id": [t.actor_rep_id for t in tasks],
            "outcome": [t.outcome.value for t in tasks],
        }
    )

    current_stage = touched["stage"].to_numpy()
    after_contact = np.where(current_stage == LeadStage.NEW.value, LeadStage.CONTACTED.value, current_stage)
    final_stage = np.where(converts, LeadStage.CONVERTED.value, after_contact)
    lead_updates = pd.DataFrame(
        {
            "lead_id": touched_lead_ids,
            "stage": final_stage,
            "updated_at_sim_day": sim_day,
        }
    )

    n_converted = int(converts.sum())
    if n_converted == 0:
        opportunities_df = _empty_frame(_OPPORTUNITY_COLUMNS)
    else:
        converted = touched.loc[converts]
        deal_size_dist = config.deal_size.resolve()
        deal_sizes = sample_positive(deal_size_dist, size=n_converted, rng=rng)
        opportunities: list[Opportunity] = [
            Opportunity(
                run_id=run_id,
                lead_id=str(lead_id),
                account_id=str(account_id),
                deal_size=float(deal_size),
                created_at_sim_day=sim_day,
                probability=config.base_close_prob,
            )
            for lead_id, account_id, deal_size in zip(
                converted["lead_id"].to_numpy(), converted["account_id"].to_numpy(), deal_sizes
            )
        ]
        opportunities_df = pd.DataFrame(
            {
                "opportunity_id": [o.opportunity_id for o in opportunities],
                "run_id": [o.run_id for o in opportunities],
                "lead_id": [o.lead_id for o in opportunities],
                "account_id": [o.account_id for o in opportunities],
                "deal_size": [o.deal_size for o in opportunities],
                "created_at_sim_day": [o.created_at_sim_day for o in opportunities],
                "stage": [o.stage.value for o in opportunities],
                "probability": [o.probability for o in opportunities],
                "expected_close_sim_day": [o.expected_close_sim_day for o in opportunities],
                "assigned_rep_id": [o.assigned_rep_id for o in opportunities],
                "closed_at_sim_day": [o.closed_at_sim_day for o in opportunities],
                "won": [o.won for o in opportunities],
            }
        )

    return ActivityResult(
        sim_day=sim_day, tasks=tasks_df, opportunities=opportunities_df, lead_updates=lead_updates
    )


def fast_forward(
    config: SimulationConfig,
    days: int,
    start_day: int = 0,
    seed: int | None = None,
    run_id: str | None = None,
    rng: np.random.Generator | None = None,
    existing_leads: pd.DataFrame | None = None,
) -> FastForwardResult:
    """Advance the simulation `days` daily ticks, starting at `start_day`: each day, new leads
    arrive (`generate_day`) and reps work the pool of open leads (`simulate_activities`), and
    return the result across all of them (see `FastForwardResult` for what's a full log vs. a
    current-state projection).

    By default, seeds a fresh `numpy.random.Generator` from `seed` (falling back to
    `config.seed` — PLAN.md section 11: "seed the RNG per run so a saved scenario can be
    reproduced or audited later"). Reusing the same seed and config reproduces byte-identical
    output.

    Pass an existing `rng` (e.g. one kept in `st.session_state` between "Fast forward" clicks) to
    *continue* a run instead: the generator draws its next values from wherever that Generator's
    stream already is, so repeated calls behave like one continuous simulation appending new days
    rather than replaying the same draws from scratch. When `rng` is given, `seed`/`config.seed`
    are ignored for this call (the Generator has already been seeded, by the caller, once).

    Pass `existing_leads` (a prior call's `.leads`) alongside a continued `rng` for the same
    reason: without it, this call's leads-current projection starts empty, so leads created by an
    earlier call would never be eligible for activities/conversion here — silently
    under-simulating a "continued" run rather than truly extending it. Continuing both together
    is also what keeps the RNG-stream-splitting property other callers rely on
    (`fast_forward(days=3); fast_forward(days=3)` sharing one `rng` matches one `fast_forward
    (days=6)` call) exact: the number of random draws `simulate_activities` makes each day
    depends on how many leads are open that day, so the two calls must see the same open-lead
    pool a single 6-day call would have.
    """
    if days < 1:
        raise ValueError("days must be >= 1.")

    if rng is None:
        rng = np.random.default_rng(seed if seed is not None else config.seed)
    resolved_run_id = run_id if run_id is not None else str(uuid4())
    account_frames: list[pd.DataFrame] = []
    task_frames: list[pd.DataFrame] = []
    opportunity_frames: list[pd.DataFrame] = []
    leads_current = existing_leads.copy() if existing_leads is not None else _empty_frame(_LEAD_COLUMNS)

    for offset in range(days):
        sim_day = start_day + offset

        day_result = generate_day(config, sim_day, rng, resolved_run_id)
        account_frames.append(day_result.accounts)
        leads_current = pd.concat([leads_current, day_result.leads], ignore_index=True)

        open_mask = ~leads_current["stage"].isin(_TERMINAL_LEAD_STAGES)
        open_leads = leads_current.loc[open_mask, ["lead_id", "account_id", "stage"]]
        activity_result = simulate_activities(config, sim_day, rng, resolved_run_id, open_leads)
        task_frames.append(activity_result.tasks)
        opportunity_frames.append(activity_result.opportunities)

        if not activity_result.lead_updates.empty:
            updates = activity_result.lead_updates.set_index("lead_id")
            leads_current = leads_current.set_index("lead_id")
            leads_current.loc[updates.index, "stage"] = updates["stage"]
            leads_current.loc[updates.index, "updated_at_sim_day"] = updates["updated_at_sim_day"]
            leads_current = leads_current.reset_index()[_LEAD_COLUMNS]

    accounts = pd.concat(account_frames, ignore_index=True) if account_frames else _empty_frame(_ACCOUNT_COLUMNS)
    tasks = pd.concat(task_frames, ignore_index=True) if task_frames else _empty_frame(_TASK_COLUMNS)
    opportunities = (
        pd.concat(opportunity_frames, ignore_index=True) if opportunity_frames else _empty_frame(_OPPORTUNITY_COLUMNS)
    )

    return FastForwardResult(
        days=days,
        start_day=start_day,
        run_id=resolved_run_id,
        accounts=accounts,
        leads=leads_current,
        tasks=tasks,
        opportunities=opportunities,
    )
