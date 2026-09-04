"""Configure Parameters -- one page, cleanly organized by entity, for every statistical
parameter this app currently exposes (Phase 1 UI feedback: "probably its own full page ... I
want to see selections for each object, cleanly organized").

Layout note (second round of feedback): related pickers sit side by side in a 2-column grid --
Leads next to Deal size, Accounts next to Close probability -- with matching elements (name,
distribution selection, parameter selection, preview chart) on the same row in each column. A
single-parameter family (e.g. Poisson) leaves a blank spacer where its second parameter would
go (see `widgets._slider_placeholder`) rather than shortening its column, so the preview charts
below stay lined up between columns.

Streamlit auto-discovers this as a page because it lives in `app/pages/`, a sibling of the entry
script (`app/main.py`). The leading `1_` controls its position in the sidebar nav; Streamlit
strips numeric prefixes and underscores when it renders the label.
"""

from __future__ import annotations

from dataclasses import replace

import streamlit as st

from funnel_sim.app.widgets import (
    deal_size_picker,
    industry_mix_picker,
    inject_placeholder_css,
    lead_arrival_picker,
    lead_conversion_picker,
    pct_slider,
    rep_activity_picker,
    task_type_mix_picker,
)
from funnel_sim.simulation.config import default_config

st.set_page_config(page_title="Configure Parameters -- Sales Funnel Simulator", layout="wide")
st.title("Configure Parameters")
st.caption(
    "Every statistical parameter below is the *true* value a later fitted model should recover "
    "(PLAN.md section 1) -- not a guess you're giving the app, but the ground truth you're "
    "setting for it to generate from."
)

if "config" not in st.session_state:
    st.session_state.config = default_config()
config = st.session_state.config

# Must run before either st.columns() block below that might contain a single-parameter
# family's placeholder slider -- see inject_placeholder_css's docstring for why this can't be
# called from inside the column (or inside the picker) instead.
inject_placeholder_css()

leads_col, deal_size_col = st.columns(2)
with leads_col:
    lead_arrival = lead_arrival_picker(config.lead_arrival)
with deal_size_col:
    deal_size = deal_size_picker(config.deal_size)

st.divider()

accounts_col, close_prob_col = st.columns(2)
with accounts_col:
    industry_mix = industry_mix_picker(config.industry_mix)
with close_prob_col:
    st.subheader("Close probability")
    base_close_prob = pct_slider("Base close probability", config.base_close_prob, key="base_close_prob")

st.divider()

activity_type_col, rep_activity_col = st.columns(2)
with activity_type_col:
    activity_type_mix = task_type_mix_picker(config.activity_type_mix)
with rep_activity_col:
    activity_prob = rep_activity_picker(config.activity_prob)

st.divider()

days_until_converted, base_conversion_prob, conversion_decay = lead_conversion_picker(
    config.days_until_converted, config.base_conversion_prob, config.conversion_decay
)

st.session_state.config = replace(
    config,
    lead_arrival=lead_arrival,
    industry_mix=industry_mix,
    deal_size=deal_size,
    base_close_prob=base_close_prob,
    activity_type_mix=activity_type_mix,
    activity_prob=activity_prob,
    days_until_converted=days_until_converted,
    base_conversion_prob=base_conversion_prob,
    conversion_decay=conversion_decay,
)

st.divider()
st.caption(
    "Employee count, ICP-fit score, and revenue band are still fixed internal defaults for "
    "Phase 1 -- not yet exposed here. Rep capacity limits aren't modeled yet, so activity "
    "chance applies uniformly rather than being gated by each rep's remaining capacity, and "
    "industry/rep-skill effects on close probability are still the next increment."
)
