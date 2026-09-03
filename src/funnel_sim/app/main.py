"""Streamlit entry point for the Sales Funnel Simulator (PLAN.md sections 7 and 10, Phase 1
MVP). Lets you set the statistical parameters that drive lead generation, fast-forward the
simulation, and see what came out of it -- the "beginnings" of the app described in Phase 1:
core entities, tunable stats parameters, a basic fast-forward, and basic charts.

Run with: `streamlit run src/funnel_sim/app/main.py`

Scope, honestly stated (see simulation/generation.py's module docstring for the same note on
the engine side): this fast-forwards Accounts and Leads only, held in `st.session_state` for the
session -- no Postgres write path yet (PLAN.md section 5's bulk-COPY loader is the next
increment). Everything below only imports from `simulation/`, never the other way around
(PLAN.md section 13).
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import plotly.express as px
import streamlit as st

from funnel_sim.simulation.config import (
    COUNT_FAMILIES,
    POSITIVE_CONTINUOUS_FAMILIES,
    ParamSpec,
    SimulationConfigError,
    default_config,
)
from funnel_sim.simulation.distributions import (
    DISCRETE_FAMILIES,
    DistributionConfigError,
    Family,
    moments_to_params,
    preview_xy,
)
from funnel_sim.simulation.generation import fast_forward

st.set_page_config(page_title="Sales Funnel Simulator", layout="wide")
st.title("Sales Funnel Simulator")
st.caption(
    "Phase 1 MVP -- configure the generative process, fast-forward, and see what it produces. "
    "Every parameter below is the *true* value a later fitted model should recover (PLAN.md section 1)."
)

if "config" not in st.session_state:
    st.session_state.config = default_config()
if "result" not in st.session_state:
    st.session_state.result = None

config = st.session_state.config


def _distribution_picker(label: str, families: tuple[Family, ...], spec, key_prefix: str):
    """One reusable (family, mean, variance) picker with a live pdf/pmf preview (PLAN.md
    section 7's "distribution picker widget"). Returns a new ParamSpec-shaped tuple
    (family, mean, variance) reflecting the widgets' current values; does not itself validate --
    the caller resolves it and handles DistributionConfigError.
    """
    st.subheader(label)
    options = [f.value for f in families]
    family_value = st.selectbox(
        "Distribution",
        options,
        index=options.index(spec.family.value) if spec.family.value in options else 0,
        key=f"{key_prefix}_family",
    )
    family = Family(family_value)
    mean = st.number_input(
        "Mean", value=float(spec.mean), step=max(abs(spec.mean) * 0.1, 0.1), key=f"{key_prefix}_mean"
    )
    variance = st.number_input(
        "Variance",
        value=float(spec.variance) if spec.variance is not None else max(mean, 1.0),
        step=max(abs(spec.mean) * 0.1, 0.1),
        key=f"{key_prefix}_variance",
    )

    try:
        resolved = moments_to_params(family, mean, variance)
        x, y = preview_xy(resolved)
        chart_df = pd.DataFrame({"x": x, "y": y})
        fig = (
            px.bar(chart_df, x="x", y="y")
            if family in DISCRETE_FAMILIES
            else px.area(chart_df, x="x", y="y")
        )
        fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig, width='stretch')
        for warning in resolved.warnings:
            st.info(warning)
    except DistributionConfigError as exc:
        st.error(str(exc))

    return family, mean, variance


with st.sidebar:
    lead_family, lead_mean, lead_variance = _distribution_picker(
        "Lead arrival (leads/day)", COUNT_FAMILIES, config.lead_arrival, "lead_arrival"
    )
    st.divider()
    deal_family, deal_mean, deal_variance = _distribution_picker(
        "Deal size ($)", POSITIVE_CONTINUOUS_FAMILIES, config.deal_size, "deal_size"
    )
    st.divider()

    base_close_prob = st.slider(
        "Base close probability", min_value=0.01, max_value=0.99, value=config.base_close_prob, step=0.01
    )

    config = replace(
        config,
        lead_arrival=ParamSpec(lead_family, lead_mean, lead_variance),
        deal_size=ParamSpec(deal_family, deal_mean, deal_variance),
        base_close_prob=base_close_prob,
    )
    st.session_state.config = config

    st.divider()
    st.subheader("Fast forward")
    days = st.number_input("Days to simulate", min_value=1, max_value=365, value=30, step=1)
    if st.button("Fast forward", type="primary", width='stretch'):
        try:
            config.validate()
            st.session_state.result = fast_forward(config, days=int(days), seed=config.seed)
        except (SimulationConfigError, DistributionConfigError) as exc:
            st.error(f"Can't fast-forward with the current parameters: {exc}")

result = st.session_state.result

if result is None:
    st.info("Set parameters in the sidebar and click **Fast forward** to generate data.")
else:
    n_leads = len(result.leads)
    n_accounts = len(result.accounts)
    col1, col2, col3 = st.columns(3)
    col1.metric("Simulated days", result.days)
    col2.metric("Accounts generated", n_accounts)
    col3.metric("Leads generated", n_leads)

    if n_leads == 0:
        st.warning("No leads were generated -- try a higher mean for lead arrival.")
    else:
        leads_per_day = (
            result.leads.groupby("created_at_sim_day").size().reset_index(name="leads")
        )
        st.plotly_chart(
            px.bar(
                leads_per_day,
                x="created_at_sim_day",
                y="leads",
                title="Leads generated per simulated day",
                labels={"created_at_sim_day": "Simulated day", "leads": "Leads"},
            ),
            width='stretch',
        )

        chart_col, table_col = st.columns(2)
        with chart_col:
            by_industry = result.leads.merge(
                result.accounts[["account_id", "industry"]], on="account_id"
            )
            industry_counts = by_industry.groupby("industry").size().reset_index(name="leads")
            st.plotly_chart(
                px.pie(industry_counts, names="industry", values="leads", title="Leads by industry"),
                width='stretch',
            )
        with table_col:
            st.caption("Sample of generated leads")
            st.dataframe(result.leads.head(50), width='stretch', height=360)
