"""funnel_sim — sales funnel simulator core package.

Submodules (see PLAN.md for the full rationale):
  simulation  — entities, effects system, distributions, fast-forward engine (UI-agnostic)
  storage     — Postgres event log, projection tables, bulk loading, DuckDB analytics access
  analysis    — regression / random forest, significance highlighting, rep performance
  app         — Streamlit UI; imports the above, never the other way around
"""

__version__ = "0.1.0"
