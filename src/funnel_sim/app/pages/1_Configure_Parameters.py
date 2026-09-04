"""Configure Parameters -- one page, cleanly organized by entity, for every statistical
parameter this app currently exposes (Phase 1 UI feedback: "probably its own full page ... I
want to see selections for each object, cleanly organized").

Streamlit auto-discovers this as a page because it lives in `app/pages/`, a sibling of the entry
script (`app/main.py`). The leading `1_` controls its position in the sidebar nav; Streamlit
strips numeric prefixes and underscores when it renders the label.
"""

from __future__ import annotations

from dataclasses import replace

import streamlit as st

from funnel_sim.app.widgets import deal_size_picker, industry_mix_picker, lead_arrival_picker, pct_slider
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

st.header("Leads")
lead_arrival = lead_arrival_picker(config.lead_arrival)

st.divider()
st.header("Accounts")
industry_mix = industry_mix_picker(config.industry_mix)

st.divider()
st.header("Deal size")
deal_size = deal_size_picker(config.deal_size)

st.divider()
st.header("Close probability")
base_close_prob = pct_slider("Base close probability", config.base_close_prob, key="base_close_prob")

st.session_state.config = replace(
    config,
    lead_arrival=lead_arrival,
    industry_mix=industry_mix,
    deal_size=deal_size,
    base_close_prob=base_close_prob,
)

st.divider()
st.caption(
    "Employee count, ICP-fit score, and revenue band are still fixed internal defaults for "
    "Phase 1 -- not yet exposed here. Task/Opportunity generation (and the effects that would "
    "make deal size and close probability actually feed into generated data) is the next "
    "increment."
)
