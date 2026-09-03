# Sales Funnel Simulator

Randomly generates sales funnel data (leads, tasks, opportunities, accounts, rep quotas) under
user-tunable statistical parameters, then automatically runs regression and random forest analysis
against the simulated history.

See [`PLAN.md`](./PLAN.md) for the full technical plan: purpose, data model, simulation engine
design, storage/scale approach, ML layer, tech stack, hosting/cost estimate, phased roadmap, and
repo structure rationale.

## Layout

```
src/funnel_sim/
  simulation/   entities, effects system, distributions, fast-forward engine (UI-agnostic)
  storage/      Postgres event log, projection tables, bulk loading, DuckDB analytics access
  analysis/     regression / random forest, significance highlighting, rep performance
  app/          Streamlit UI — imports the above, never the other way around
notebooks/      Phase 0 prototyping (kept out of the app's dependency path)
migrations/     Postgres schema migrations
tests/
scripts/        maintenance scripts (retention/cleanup job, seed data, etc.)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Status

- **Phase 0 (done):** effects system (logit-space composition) and distribution parameterization
  (method-of-moments) prototyped and unit-tested in `simulation/distributions.py` and
  `simulation/effects.py`.
- **Phase 1 (in progress):** MVP.
  - Core data objects for every entity in PLAN.md section 3 (`simulation/entities.py`).
  - Statistical parameter configuration built on Phase 0 (`simulation/config.py`) — pick a
    distribution family, mean, and variance for lead arrival and deal size; set industry mix,
    base close probability, and per-industry/per-rep effects.
  - A vectorized fast-forward generator (`simulation/generation.py`) — currently produces
    Accounts and Leads; Task/Opportunity generation through the effects system is next.
  - A Streamlit app (`app/main.py`) with a live distribution-picker sidebar and basic charts —
    run it with `streamlit run src/funnel_sim/app/main.py`.
  - A Postgres schema migration (`migrations/0001_initial_schema.sql`) for the event-sourcing
    model, applied via `storage/schema.py` — not yet wired to the generator (no bulk-COPY write
    path yet).

See PLAN.md's phased roadmap (section 10) for the rest of what's next.
