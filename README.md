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

Scaffold only — see PLAN.md's phased roadmap (section 10) for what's next (Phase 0: prototype the
effects system and distribution parameterization before building app code).
