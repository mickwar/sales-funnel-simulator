"""Core data objects for the sales funnel (PLAN.md section 3).

These dataclasses are the entity model the rest of the simulation, storage, and analysis
layers build on: Account -> Lead -> Task/Opportunity, plus Rep, RepQuota, and Run. Every entity
carries `run_id` so the data model is multi-tenant from day one (PLAN.md section 5) — one row
in Postgres, filtered by `run_id`, rather than a per-user table.

This is Phase 1's "beginnings of the core data objects" (PLAN.md section 10): every entity from
the plan is represented here, but the fast-forward generator (`generation.py`) currently only
populates Account and Lead. Task, Opportunity, Rep, and RepQuota generation land in the next
Phase 1 increment.

Instances represent *current state* — the row you'd find in a `<entity>_current` projection
table (PLAN.md section 3), not an event. `sim_day` fields are ticks in the simulation's own
clock (PLAN.md section 4's daily-tick default), not wall-clock timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4


def _new_id() -> str:
    """A fresh entity id. Plain str (not uuid.UUID) so entities serialize to JSON/DataFrames
    without extra conversion — Postgres columns are still UUID-typed (see migrations/).
    """
    return str(uuid4())


class Industry(str, Enum):
    """A small, fixed set of ICP archetypes (PLAN.md section 12 flags this as an open question —
    start with a fixed set, revisit if the app needs fully user-defined industries later).
    """

    SAAS = "saas"
    MANUFACTURING = "manufacturing"
    HEALTHCARE = "healthcare"
    FINANCIAL_SERVICES = "financial_services"
    RETAIL = "retail"
    OTHER = "other"


class RevenueBand(str, Enum):
    """Coarse annual-revenue buckets for an Account, used as a segmentation/effect axis."""

    UNDER_1M = "under_1m"
    FROM_1M_TO_10M = "1m_10m"
    FROM_10M_TO_100M = "10m_100m"
    OVER_100M = "over_100m"


class LeadStage(str, Enum):
    """Lifecycle stage of a Lead. CONVERTED means an Opportunity has been created from it."""

    NEW = "new"
    CONTACTED = "contacted"
    QUALIFIED = "qualified"
    DISQUALIFIED = "disqualified"
    CONVERTED = "converted"


class OpportunityStage(str, Enum):
    """Pipeline stage of an Opportunity (PLAN.md section 3: "stage, probability, deal size,
    expected close date").
    """

    PROSPECTING = "prospecting"
    QUALIFICATION = "qualification"
    PROPOSAL = "proposal"
    NEGOTIATION = "negotiation"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"


class TaskType(str, Enum):
    """Activity types a Rep performs (PLAN.md section 4: "call, email, meeting, text")."""

    CALL = "call"
    EMAIL = "email"
    MEETING = "meeting"
    TEXT = "text"


class TaskOutcome(str, Enum):
    """Result of a Task, once performed. PENDING is the state before it's been carried out."""

    PENDING = "pending"
    NO_RESPONSE = "no_response"
    CONNECTED = "connected"
    ADVANCED = "advanced"


@dataclass
class Account:
    """A company (PLAN.md section 3)."""

    run_id: str
    industry: Industry
    employee_count: int
    revenue_band: RevenueBand
    icp_fit_score: float  # 0-1; higher means a better fit for the ideal customer profile.
    name: str = ""
    created_at_sim_day: int = 0
    account_id: str = field(default_factory=_new_id)


@dataclass
class Lead:
    """An individual/contact tied to an Account, with a lifecycle stage (PLAN.md section 3).

    `days_until_converted`, `base_conversion_prob`, `conversion_decay`, and `deal_size` are all
    drawn once, when the Lead is generated (see `generation.generate_day`) -- not derived from
    any later activity. `conversion_prob` is the *current* probability, which starts equal to
    `base_conversion_prob` and then evolves daily: decayed multiplicatively by
    `conversion_decay` each day the Lead doesn't convert, and nudged (positively or negatively)
    by whatever `task_type_effects` apply when a rep activity happens to it (PLAN.md/Phase 1
    feedback: "the chance of a lead being converted should not depend on the outcome of any
    particular activity[, but] activities... can affect the probability"). A Lead that reaches
    `days_until_converted` days old without converting moves to DISQUALIFIED.
    """

    run_id: str
    account_id: str
    created_at_sim_day: int
    stage: LeadStage = LeadStage.NEW
    source: str = "unknown"  # e.g. "inbound", "outbound", "referral" — used as a confounder in
    # rep-performance adjustment later (PLAN.md section 6: "stratify by ICP-fit/lead source").
    assigned_rep_id: str | None = None
    updated_at_sim_day: int = 0
    days_until_converted: int = 0  # 0 means it must convert (or be disqualified) the same day.
    base_conversion_prob: float = 0.0  # this Lead's starting conversion probability (0-1).
    conversion_decay: float = 1.0  # daily multiplicative decay applied while unconverted (0-1).
    conversion_prob: float = 0.0  # this Lead's *current* conversion probability (0-1) -- starts
    # equal to base_conversion_prob, then decays/shifts daily (see class docstring).
    deal_size: float = 0.0  # this Lead's deal size if/when it converts into an Opportunity --
    # drawn once at creation so the same lead always carries the same potential deal size.
    lead_id: str = field(default_factory=_new_id)


@dataclass
class Rep:
    """A salesperson: capacity, assigned quota, tenure/skill parameters (PLAN.md section 3)."""

    run_id: str
    name: str
    capacity_tasks_per_day: int
    tenure_days: int = 0
    # The *true* configured skill effect for this rep, in logit space (PLAN.md sections 1/4) —
    # what a later fitted model's per-rep coefficient should be compared against.
    skill_logit_delta: float = 0.0
    rep_id: str = field(default_factory=_new_id)


@dataclass
class RepQuota:
    """A quota period, target, and attainment for a Rep (PLAN.md section 3)."""

    run_id: str
    rep_id: str
    period_start_sim_day: int
    period_end_sim_day: int
    target: float
    attained: float = 0.0
    quota_id: str = field(default_factory=_new_id)


@dataclass
class Task:
    """An activity a rep performs, tied to a Lead or an Opportunity (PLAN.md section 3)."""

    run_id: str
    task_type: TaskType
    sim_day: int
    lead_id: str | None = None
    opportunity_id: str | None = None
    actor_rep_id: str | None = None
    outcome: TaskOutcome = TaskOutcome.PENDING
    task_id: str = field(default_factory=_new_id)

    def __post_init__(self) -> None:
        if self.lead_id is None and self.opportunity_id is None:
            raise ValueError("Task must be tied to a lead_id or an opportunity_id.")


@dataclass
class Opportunity:
    """A deal in progress: stage, probability, deal size, expected close date (PLAN.md
    section 3).
    """

    run_id: str
    lead_id: str
    account_id: str
    deal_size: float
    created_at_sim_day: int
    stage: OpportunityStage = OpportunityStage.PROSPECTING
    probability: float = 0.1
    expected_close_sim_day: int | None = None
    assigned_rep_id: str | None = None
    closed_at_sim_day: int | None = None
    won: bool | None = None  # None until closed; True/False once CLOSED_WON/CLOSED_LOST.
    opportunity_id: str = field(default_factory=_new_id)


@dataclass
class Run:
    """One named parameter set + its simulated history (PLAN.md section 3) — what makes "save
    scenario A vs B" possible later (PLAN.md section 7).
    """

    name: str
    seed: int
    status: str = "active"
    created_at_sim_day: int = 0
    run_id: str = field(default_factory=_new_id)
