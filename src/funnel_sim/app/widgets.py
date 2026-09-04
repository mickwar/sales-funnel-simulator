"""Reusable Streamlit widgets for the "Configure Parameters" page (Phase 1 UI feedback round).

Streamlit widget-state note, since every picker below leans on it: once a widget is instantiated
with an explicit `key=`, that widget reads and writes `st.session_state[key]` directly on every
rerun -- the `value=` argument (if any) is only honored the very first time the key appears.
That means the *only* way to programmatically change a widget's current value (e.g. "auto-bump
Negative Binomial's variance slider when the mean slider is dragged past it") is to mutate
`st.session_state[key]` *before* the widget with that key is instantiated on this rerun. Every
"auto-adjust" / clamp below follows that pattern: touch `st.session_state[key]`, then call the
`st.slider(...)`/`st.selectbox(...)` that owns it.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from funnel_sim.simulation.config import (
    COUNT_FAMILIES,
    DEAL_SIZE_FAMILIES,
    ParamSpec,
    SimulationConfigError,
    validate_deal_size_spec,
)
from funnel_sim.simulation.distributions import (
    DISCRETE_FAMILIES,
    SINGLE_PARAMETER_FAMILIES,
    DistributionConfigError,
    moments_to_params,
    preview_xy,
)
from funnel_sim.simulation.entities import Industry

from .formatting import humanize


def inject_placeholder_css() -> None:
    """One-time, *page-level* CSS injection that hides every `_slider_placeholder` widget below,
    matched by a shared substring in their `key` (`_variance_placeholder`) rather than one exact
    key, since more than one picker column can each render its own placeholder.

    Must be called once from the page itself, before any `st.columns(...)` block that will
    contain a placeholder -- NOT from inside `_slider_placeholder`. The first (pixel-height)
    version of this fix injected the `<style>` tag via its own `st.markdown(...)` call sitting
    right next to the hidden slider, inside that slider's own column -- which reserved the right
    *height* but still misaligned the columns, because that markdown call is itself an extra
    flex child in Streamlit's column layout, adding one extra inter-widget gap (~16px) to that
    column alone that the sibling column never had. Injecting the rule once, above both columns,
    keeps every column's child count (and therefore its gaps) identical.

    The rule targets both the placeholder's container *and* every element inside it (`... *`),
    with `!important` on both: Streamlit's own stylesheet explicitly sets its widget label back
    to `visibility: visible` (for accessibility), which -- being a rule on the label itself, not
    inherited from its hidden ancestor -- otherwise wins and leaves a stray "Variance" label
    showing above the blank space.
    """
    st.markdown(
        "<style>"
        "[class*='_variance_placeholder'], [class*='_variance_placeholder'] * "
        "{ visibility: hidden !important; }"
        "</style>",
        unsafe_allow_html=True,
    )


def _slider_placeholder(key: str) -> None:
    """Reserve *exactly* the vertical footprint an `st.slider` takes here, so a single-parameter
    family (e.g. Poisson, with no variance control) doesn't leave its column shorter than a
    two-parameter family's -- keeping side-by-side columns' rows (and, in turn, their preview
    charts) lined up (Phase 1 feedback: "add a blank space for where a second parameter would
    normally go").

    A hardcoded pixel height drifts out of sync with the real slider it's supposed to match
    (label wrapping, theme, Streamlit version). Instead, render an actual disabled `st.slider`
    -- guaranteed identical layout to a real one -- hidden with `visibility: hidden` rather than
    `display: none` (which would collapse the space instead of reserving it) by the page-level
    `inject_placeholder_css()` above. Passing `key=` gives the widget's wrapper a stable
    `st-key-<key>` CSS class (Streamlit feature) for that rule to match.
    """
    st.slider("Variance", min_value=0.0, max_value=1.0, value=0.0, key=key, disabled=True)


def _render_preview(resolved, lower_bound: float | None = None) -> None:
    """Render the pdf/pmf preview chart right next to the parameter selection -- users called
    this out as the thing they want to see more of.
    """
    try:
        x, y = preview_xy(resolved, lower_bound=lower_bound)
    except DistributionConfigError as exc:
        st.error(str(exc))
        return
    chart_df = pd.DataFrame({"x": x, "y": y})
    fig = (
        px.bar(chart_df, x="x", y="y")
        if resolved.family in DISCRETE_FAMILIES
        else px.area(chart_df, x="x", y="y")
    )
    fig.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    for warning in resolved.warnings:
        st.info(warning)


def lead_arrival_picker(spec: ParamSpec, key_prefix: str = "lead_arrival") -> ParamSpec:
    """Family + mean (+ variance, unless the family is single-parameter, e.g. Poisson) picker
    for the daily lead-arrival count process.
    """
    st.subheader("Lead arrival (leads / day)")

    family_key = f"{key_prefix}_family"
    if family_key not in st.session_state:
        st.session_state[family_key] = spec.family
    family = st.selectbox(
        "Distribution",
        list(COUNT_FAMILIES),
        format_func=lambda f: humanize(f.value),
        key=family_key,
        help=(
            "Poisson has one free parameter (variance always equals the mean). Negative "
            "Binomial adds an independent variance control for over-dispersed counts."
        ),
    )

    mean_key = f"{key_prefix}_mean"
    if mean_key not in st.session_state:
        st.session_state[mean_key] = float(np.clip(spec.mean, 0.1, 500.0))
    st.session_state[mean_key] = float(np.clip(st.session_state[mean_key], 0.1, 500.0))
    mean = st.slider("Mean", min_value=0.1, max_value=500.0, step=0.5, key=mean_key)

    variance: float | None
    if family in SINGLE_PARAMETER_FAMILIES:
        # Poisson has only one free parameter -- showing a variance control would imply it's
        # independently settable, which it isn't (Phase 1 feedback: "only give it the one
        # parameter"). Leave a blank spacer in its place so this column's rows still line up
        # with a two-parameter family's (e.g. deal size) in a side-by-side layout.
        variance = None
        _slider_placeholder(f"{key_prefix}_variance_placeholder")
    else:
        var_key = f"{key_prefix}_variance"
        # Negative Binomial requires variance > mean, so mean + epsilon is the lowest legal
        # variance; $1000 is a fixed hard cap on the high end -- only the lower bound moves as
        # the mean slider moves (Phase 1 feedback: "put a hard cap of 1000 ... only allow the
        # lower bound to change").
        min_variance = mean + 0.01
        max_variance = 1000.0
        if var_key not in st.session_state:
            default_variance = spec.variance if (spec.variance and spec.variance > spec.mean) else mean * 2 + 1.0
            st.session_state[var_key] = float(np.clip(default_variance, min_variance, max_variance))
        # If the mean slider was just dragged past the stored variance (or the fixed cap now
        # sits below it), the current value is no longer valid -- snap it to the new lowest
        # legal value. Otherwise leave whatever value the user already chose untouched (Phase 1
        # feedback: "variance stays the same unless it became invalid, in which case update it
        # to be the lowest legal value").
        if st.session_state[var_key] <= mean:
            st.session_state[var_key] = min_variance
        st.session_state[var_key] = float(np.clip(st.session_state[var_key], min_variance, max_variance))
        variance = st.slider(
            "Variance", min_value=min_variance, max_value=max_variance, step=0.5, key=var_key
        )

    try:
        resolved = moments_to_params(family, mean, variance)
        _render_preview(resolved)
    except DistributionConfigError as exc:
        st.error(str(exc))

    return ParamSpec(family, mean, variance)


def deal_size_picker(spec: ParamSpec, key_prefix: str = "deal_size") -> ParamSpec:
    """Family + mean + standard-deviation picker for deal size, with the $0 floor enforced at
    generation time via rejection sampling (`distributions.sample_positive`) -- previewed here
    with the matching truncated-and-renormalized density (`lower_bound=0.0`).
    """
    st.subheader("Deal size ($)")

    family_key = f"{key_prefix}_family"
    if family_key not in st.session_state:
        st.session_state[family_key] = spec.family
    family = st.selectbox(
        "Distribution",
        list(DEAL_SIZE_FAMILIES),
        format_func=lambda f: humanize(f.value),
        key=family_key,
        help=(
            "Deal size has a hard $0 floor: Gamma/Lognormal are positive by construction, and "
            "Normal enforces it by redrawing any value that lands at or below $0."
        ),
    )

    # Hard $0 floor, structurally: the mean slider itself never offers a value at or below $0 --
    # for any of the three families, not just Normal -- rather than allowing the selection and
    # catching it with an error afterward (Phase 1 feedback: "Don't let the user even select
    # below that").
    mean_key = f"{key_prefix}_mean"
    if mean_key not in st.session_state:
        st.session_state[mean_key] = float(np.clip(spec.mean, 50.0, 10_000.0))
    st.session_state[mean_key] = float(np.clip(st.session_state[mean_key], 50.0, 10_000.0))
    mean = st.slider(
        "Mean", min_value=50.0, max_value=10_000.0, step=50.0, format="$%.0f", key=mean_key
    )

    std_key = f"{key_prefix}_std"
    default_std = float(np.sqrt(spec.variance)) if spec.variance else 200.0
    if std_key not in st.session_state:
        st.session_state[std_key] = float(np.clip(default_std, 1.0, 500.0))
    st.session_state[std_key] = float(np.clip(st.session_state[std_key], 1.0, 500.0))
    std = st.slider(
        "Standard deviation",
        min_value=1.0,
        max_value=500.0,
        step=1.0,
        format="$%.0f",
        key=std_key,
    )
    variance = std**2

    try:
        validate_deal_size_spec(ParamSpec(family, mean, variance))
        resolved = moments_to_params(family, mean, variance)
        _render_preview(resolved, lower_bound=0.0)
    except (SimulationConfigError, DistributionConfigError) as exc:
        st.error(str(exc))

    return ParamSpec(family, mean, variance)


def industry_mix_picker(
    industry_mix: Mapping[Industry, float], key_prefix: str = "industry_mix"
) -> dict[Industry, float]:
    """Percentage-per-industry sliders (Phase 1 feedback: percentages, not raw probabilities),
    auto-normalized to sum to 100% so the caller always gets a valid mix back.
    """
    st.subheader("Industry mix")
    st.caption("Share of newly-generated accounts drawn from each industry.")

    industries = list(Industry)
    raw: dict[Industry, float] = {}
    cols = st.columns(2)
    for i, industry in enumerate(industries):
        default_pct = industry_mix.get(industry, 1.0 / len(industries)) * 100.0
        pct_key = f"{key_prefix}_{industry.value}_pct"
        if pct_key not in st.session_state:
            st.session_state[pct_key] = float(np.clip(round(default_pct, 1), 0.0, 100.0))
        with cols[i % 2]:
            raw[industry] = st.slider(
                humanize(industry.value),
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                format="%.0f%%",
                key=pct_key,
            )

    raw_total = sum(raw.values())
    if raw_total <= 0:
        st.error("At least one industry needs a nonzero weight.")
        return {industry: 1.0 / len(industries) for industry in industries}

    st.caption(f"Raw total: {raw_total:.0f}% -> normalized to 100%.")
    return {industry: weight / raw_total for industry, weight in raw.items()}


def pct_slider(
    label: str,
    value: float,
    key: str,
    min_pct: float = 1.0,
    max_pct: float = 99.0,
) -> float:
    """A percent-labeled slider (default 1-99%, since a hard 0% or 100% probability isn't a real
    choice here) that returns a 0-1 fraction -- percentages read more intuitively than raw
    probabilities (Phase 1 feedback).
    """
    pct_key = f"{key}_pct"
    if pct_key not in st.session_state:
        st.session_state[pct_key] = float(np.clip(round(value * 100.0, 1), min_pct, max_pct))
    st.session_state[pct_key] = float(np.clip(st.session_state[pct_key], min_pct, max_pct))
    pct = st.slider(label, min_value=min_pct, max_value=max_pct, step=0.5, format="%.1f%%", key=pct_key)
    return pct / 100.0
