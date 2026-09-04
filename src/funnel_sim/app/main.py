"""Streamlit entry point for the Sales Funnel Simulator (PLAN.md sections 7 and 10, Phase 1
MVP): the "Run" page -- fast-forward the simulation and see what came out of it. Parameter
configuration lives on its own page (`pages/1_Configure_Parameters.py`); this page is just
Run + Reset + results, so the app opens straight to "what did the simulation produce" rather
than a wall of sliders.

Run with: `streamlit run src/funnel_sim/app/main.py`

Scope, honestly stated (see simulation/generation.py's module docstring for the same note on
the engine side): this fast-forwards Accounts and Leads only, held in `st.session_state` for the
session -- no Postgres write path yet (PLAN.md section 5's bulk-COPY loader is the next
increment). Everything below only imports from `simulation/`, never the other way around
(PLAN.md section 13).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from funnel_sim.app.formatting import humanize
from funnel_sim.simulation.config import SimulationConfigError, default_config
from funnel_sim.simulation.distributions import DistributionConfigError
from funnel_sim.simulation.generation import fast_forward

st.set_page_config(page_title="Sales Funnel Simulator", layout="wide")
st.title("Sales Funnel Simulator")
st.caption(
    "Phase 1 MVP -- fast-forward the simulation and see what it produces. Head to **Configure "
    "Parameters** in the sidebar to change the generative process."
)


def _fresh_run_state() -> dict:
    return {
        "run_id": None,
        "rng": None,
        "total_days": 0,
        "accounts": None,
        "leads": None,
    }


if "config" not in st.session_state:
    st.session_state.config = default_config()
if "run" not in st.session_state:
    st.session_state.run = _fresh_run_state()

config = st.session_state.config
run = st.session_state.run

with st.sidebar:
    st.subheader("Fast forward")
    days = st.number_input("Days to simulate", min_value=1, max_value=365, value=30, step=1)

    if st.button("Fast forward", type="primary", use_container_width=True):
        try:
            config.validate()
            # Reuse the existing rng (and run_id) across clicks so repeated "Fast forward"s
            # append new simulated days onto the same continuous run instead of overwriting it
            # with a fresh replay from day 0 (Phase 1 feedback).
            if run["rng"] is None:
                run["rng"] = np.random.default_rng(config.seed)
            result = fast_forward(
                config,
                days=int(days),
                start_day=run["total_days"],
                rng=run["rng"],
                run_id=run["run_id"],
            )
            run["run_id"] = result.run_id
            run["total_days"] += result.days
            run["accounts"] = (
                result.accounts
                if run["accounts"] is None
                else pd.concat([run["accounts"], result.accounts], ignore_index=True)
            )
            run["leads"] = (
                result.leads
                if run["leads"] is None
                else pd.concat([run["leads"], result.leads], ignore_index=True)
            )
        except (SimulationConfigError, DistributionConfigError) as exc:
            st.error(f"Can't fast-forward with the current parameters: {exc}")

    if st.button("Reset run", use_container_width=True):
        st.session_state.run = _fresh_run_state()
        run = st.session_state.run

if run["leads"] is None:
    st.info("Click **Fast forward** to generate data. Adjust parameters on the Configure Parameters page first if you like.")
else:
    accounts, leads = run["accounts"], run["leads"]
    n_leads, n_accounts = len(leads), len(accounts)
    col1, col2, col3 = st.columns(3)
    col1.metric("Simulated days", run["total_days"])
    col2.metric("Accounts generated", n_accounts)
    col3.metric("Leads generated", n_leads)

    if n_leads == 0:
        st.warning("No leads were generated -- try a higher mean for lead arrival on the Configure Parameters page.")
    else:
        leads_per_day = leads.groupby("created_at_sim_day").size().reset_index(name="leads")
        st.plotly_chart(
            px.bar(
                leads_per_day,
                x="created_at_sim_day",
                y="leads",
                title="Leads generated per simulated day",
                labels={"created_at_sim_day": "Simulated day", "leads": "Leads"},
            ),
            use_container_width=True,
        )

        chart_col, table_col = st.columns(2)
        with chart_col:
            by_industry = leads.merge(accounts[["account_id", "industry"]], on="account_id")
            industry_counts = by_industry.groupby("industry").size().reset_index(name="leads")
            industry_counts["industry"] = industry_counts["industry"].map(humanize)
            st.plotly_chart(
                px.pie(industry_counts, names="industry", values="leads", title="Leads by industry"),
                use_container_width=True,
            )
        with table_col:
            st.caption("Most recently generated leads")
            display = leads.tail(50).copy()
            display["stage"] = display["stage"].map(humanize)
            st.dataframe(display, use_container_width=True, height=360)
