"""Streamlit entry point for the Sales Funnel Simulator (PLAN.md sections 7 and 10, Phase 1
MVP): the "Run" page -- fast-forward the simulation and see what came out of it. Parameter
configuration lives on its own page (`pages/1_Configure_Parameters.py`); this page is just
Run + Reset + results, so the app opens straight to "what did the simulation produce" rather
than a wall of sliders.

Run with: `streamlit run src/funnel_sim/app/main.py`

Scope, honestly stated (see simulation/generation.py's module docstring for the same note on
the engine side): this fast-forwards Accounts, Leads, Tasks, and Opportunities, held in
`st.session_state` for the session -- no Postgres write path yet (PLAN.md section 5's bulk-COPY
loader is the next increment). Everything below only imports from `simulation/`, never the other
way around (PLAN.md section 13).
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
        "tasks": None,
        "opportunities": None,
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
                # Hand back the previous click's current-leads projection so this click's
                # activity/conversion draws see the whole open-lead pool, not just today's new
                # arrivals (see fast_forward's docstring on `existing_leads`).
                existing_leads=run["leads"],
            )
            run["run_id"] = result.run_id
            run["total_days"] += result.days
            run["accounts"] = (
                result.accounts
                if run["accounts"] is None
                else pd.concat([run["accounts"], result.accounts], ignore_index=True)
            )
            # `leads` is a current-state projection (one row per lead, PLAN.md section 3), not an
            # append-only log -- each click's result already reflects every lead ever generated,
            # so it replaces the stored value rather than being concatenated onto it.
            run["leads"] = result.leads
            run["tasks"] = (
                result.tasks
                if run["tasks"] is None
                else pd.concat([run["tasks"], result.tasks], ignore_index=True)
            )
            run["opportunities"] = (
                result.opportunities
                if run["opportunities"] is None
                else pd.concat([run["opportunities"], result.opportunities], ignore_index=True)
            )
        except (SimulationConfigError, DistributionConfigError) as exc:
            st.error(f"Can't fast-forward with the current parameters: {exc}")

    if st.button("Reset run", use_container_width=True):
        st.session_state.run = _fresh_run_state()
        run = st.session_state.run

if run["leads"] is None:
    st.info("Click **Fast forward** to generate data. Adjust parameters on the Configure Parameters page first if you like.")
else:
    accounts, leads, tasks, opportunities = run["accounts"], run["leads"], run["tasks"], run["opportunities"]
    n_leads, n_accounts, n_tasks, n_opportunities = len(leads), len(accounts), len(tasks), len(opportunities)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Simulated days", run["total_days"])
    col2.metric("Accounts generated", n_accounts)
    col3.metric("Leads generated", n_leads)
    col4.metric("Activities logged", n_tasks)
    col5.metric("Opportunities created", n_opportunities)

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
            st.caption("Leads by stage")
            stage_counts = leads.groupby("stage").size().reset_index(name="leads")
            stage_counts["stage"] = stage_counts["stage"].map(humanize)
            st.plotly_chart(
                px.bar(stage_counts, x="stage", y="leads", labels={"stage": "Stage", "leads": "Leads"}),
                use_container_width=True,
            )

        st.divider()
        st.subheader("Rep activity")
        if n_tasks == 0:
            st.info(
                "No activities were logged -- raise the activity chance on the Configure "
                "Parameters page (or fast-forward more days) to see reps work these leads."
            )
        else:
            activity_chart_col, activity_table_col = st.columns(2)
            with activity_chart_col:
                by_type = tasks.groupby("task_type").size().reset_index(name="activities")
                by_type["task_type"] = by_type["task_type"].map(humanize)
                st.plotly_chart(
                    px.pie(by_type, names="task_type", values="activities", title="Activities by type"),
                    use_container_width=True,
                )
            with activity_table_col:
                by_outcome = tasks.groupby("outcome").size().reset_index(name="activities")
                by_outcome["outcome"] = by_outcome["outcome"].map(humanize)
                st.plotly_chart(
                    px.pie(by_outcome, names="outcome", values="activities", title="Activities by outcome"),
                    use_container_width=True,
                )

        st.divider()
        st.subheader("Recently generated rows")
        leads_tab, tasks_tab, opportunities_tab = st.tabs(["Leads", "Activities", "Opportunities"])
        with leads_tab:
            display = leads.tail(50).copy()
            display["stage"] = display["stage"].map(humanize)
            st.dataframe(display, use_container_width=True, height=360)
        with tasks_tab:
            if n_tasks == 0:
                st.caption("No activities yet.")
            else:
                display = tasks.tail(50).copy()
                display["task_type"] = display["task_type"].map(humanize)
                display["outcome"] = display["outcome"].map(humanize)
                st.dataframe(display, use_container_width=True, height=360)
        with opportunities_tab:
            if n_opportunities == 0:
                st.caption("No opportunities yet.")
            else:
                display = opportunities.tail(50).copy()
                display["stage"] = display["stage"].map(humanize)
                st.dataframe(display, use_container_width=True, height=360)
