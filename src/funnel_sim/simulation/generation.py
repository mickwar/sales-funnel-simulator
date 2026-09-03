"""Vectorized, time-stepped data generation — the beginnings of the fast-forward engine
(PLAN.md section 4 and section 10 Phase 1).

Per PLAN.md section 4: "fast forward N days" advances one daily tick at a time; within a tick,
every statistical draw is generated as numpy arrays in one shot rather than looped row-by-row in
Python — that's the difference between a fast-forward that takes seconds and one that takes
minutes at the row counts this app targets (hundreds of thousands of rows/session, PLAN.md
section 5). Constructing one entity object per generated row (to get each its own id) is still a
Python-level loop below, but it does no sampling — every random draw happens once, vectorized,
before that loop runs.

Phase 1 scope, honestly stated: this module generates Accounts and Leads only, from
`SimulationConfig.lead_arrival` and `.industry_mix` (§4's Poisson/Negative-Binomial arrival
process). Task/Opportunity generation — sampling task outcomes and close probability through the
effects system (§4, §6) — is the next Phase 1 increment; `SimulationConfig.deal_size`,
`.base_close_prob`, `.industry_effects`, and `.rep_skill_effects` are defined already (see
config.py) so that next step doesn't require reshaping the config.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .entities import Account, Industry, Lead, LeadStage, RevenueBand

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


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(dtype="object") for col in columns})


@dataclass(frozen=True)
class DayResult:
    """One simulated day's worth of newly-generated entities, as DataFrames (one row per
    entity) with columns matching `entities.Account`/`entities.Lead` field names 1:1 — that
    correspondence is what makes bulk-loading these into Postgres (PLAN.md section 5) a direct
    column mapping rather than a translation layer.

    Empty (zero-lead) days still return DataFrames with the right columns, never `None` — so
    callers can always `pd.concat` a sequence of `DayResult`s without special-casing.
    """

    sim_day: int
    accounts: pd.DataFrame
    leads: pd.DataFrame


@dataclass(frozen=True)
class FastForwardResult:
    """The concatenated output of fast-forwarding a SimulationConfig for a number of days.

    `run_id` is the id every generated Account/Lead row carries (PLAN.md section 5's
    multi-tenancy column) — a fresh one is minted per call unless the caller passes one in
    (e.g. to keep generating into an existing Run/scenario).
    """

    days: int
    start_day: int
    run_id: str
    accounts: pd.DataFrame
    leads: pd.DataFrame


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


def fast_forward(
    config: SimulationConfig,
    days: int,
    start_day: int = 0,
    seed: int | None = None,
    run_id: str | None = None,
) -> FastForwardResult:
    """Advance the simulation `days` daily ticks, starting at `start_day`, and return the
    concatenated Accounts/Leads generated across all of them.

    Seeds a fresh `numpy.random.Generator` from `seed` (falling back to `config.seed` — PLAN.md
    section 11: "seed the RNG per run so a saved scenario can be reproduced or audited later").
    Reusing the same seed and config reproduces byte-identical output.
    """
    if days < 1:
        raise ValueError("days must be >= 1.")

    rng = np.random.default_rng(seed if seed is not None else config.seed)
    resolved_run_id = run_id if run_id is not None else str(uuid4())
    account_frames: list[pd.DataFrame] = []
    lead_frames: list[pd.DataFrame] = []

    for offset in range(days):
        result = generate_day(config, start_day + offset, rng, resolved_run_id)
        account_frames.append(result.accounts)
        lead_frames.append(result.leads)

    accounts = pd.concat(account_frames, ignore_index=True) if account_frames else _empty_frame(_ACCOUNT_COLUMNS)
    leads = pd.concat(lead_frames, ignore_index=True) if lead_frames else _empty_frame(_LEAD_COLUMNS)

    return FastForwardResult(
        days=days, start_day=start_day, run_id=resolved_run_id, accounts=accounts, leads=leads
    )
