"""Reusable Streamlit widgets for the "Configure Parameters" page (Phase 1 UI feedback round).

Streamlit widget-state note, since every picker below leans on it: once a widget is instantiated
with an explicit `key=`, that widget reads and writes `st.session_state[key]` directly on every
rerun -- the `value=` argument (if any) is only honored the very first time the key appears.
That means the *only* way to programmatically change a widget's current value (e.g. "auto-bump
Negative Binomial's SD slider when the mean slider is dragged past it") is to mutate
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
    UNIFORM_FAMILIES,
    DistributionConfigError,
    Family,
    preview_xy,
)
from funnel_sim.simulation.entities import Industry, TaskType

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
    family (e.g. Poisson, with no second control) doesn't leave its column shorter than a
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
    """Family + a second control (variance/SD, low/high, or nothing) picker for the daily
    lead-arrival count process. Poisson has just the one Mean slider; Negative Binomial adds a
    Standard Deviation slider; Discrete Uniform replaces Mean entirely with Low/High endpoint
    sliders (Phase 1 feedback: "the two parameters should be the left and right endpoints").
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
            "Binomial adds an independent standard-deviation control for over-dispersed counts. "
            "Discrete Uniform makes every whole number in a range equally likely."
        ),
    )

    mean: float | None = None
    variance: float | None = None
    low: float | None = None
    high: float | None = None

    if family in UNIFORM_FAMILIES:
        # Lead count is a whole-number-per-day quantity, so both endpoints are integer-stepped,
        # bounded 0-100 (Phase 1 feedback: "the lowest value can be 0 and the highest value is
        # 100") -- only the low endpoint's own slider can reach 0; Poisson/Negative Binomial's
        # Mean still floors at 1 since their means must be strictly positive.
        low_key = f"{key_prefix}_low"
        if low_key not in st.session_state:
            default_low = spec.low if spec.low is not None else 1
            st.session_state[low_key] = int(np.clip(round(default_low), 0, 99))
        # Low's own slider tops out one below the overall 100 cap -- it and High must be
        # distinct whole numbers, and Streamlit's slider rejects a min_value == max_value, so
        # High always needs at least one point of headroom above Low (see the min/max clamp
        # below for how a Low of 99 still resolves to a valid High of exactly 100).
        st.session_state[low_key] = int(np.clip(st.session_state[low_key], 0, 99))
        low = st.slider("Low", min_value=0, max_value=99, step=1, key=low_key)

        true_min_high = low + 1  # the mathematically correct lower bound (may be 100 itself).
        slider_min_high = min(true_min_high, 99)  # keeps High's own min_value < max_value=100.
        high_key = f"{key_prefix}_high"
        if high_key not in st.session_state:
            default_high = spec.high if spec.high is not None else 100
            st.session_state[high_key] = int(np.clip(round(default_high), true_min_high, 100))
        if st.session_state[high_key] < true_min_high:
            st.session_state[high_key] = true_min_high
        st.session_state[high_key] = int(np.clip(st.session_state[high_key], slider_min_high, 100))
        high = st.slider("High", min_value=slider_min_high, max_value=100, step=1, key=high_key)
    else:
        # Lead count is a whole-number-per-day quantity, so the Mean control is itself
        # integer-stepped, bounded to a realistic 1-100 leads/day range (Phase 1 feedback: "Set
        # the lead count boundaries for Mean to be between 1 and 100, enforce whole numbers").
        mean_key = f"{key_prefix}_mean"
        if mean_key not in st.session_state:
            default_mean = spec.mean if spec.mean is not None else 20
            st.session_state[mean_key] = int(np.clip(round(default_mean), 1, 100))
        st.session_state[mean_key] = int(np.clip(st.session_state[mean_key], 1, 100))
        mean = st.slider("Mean", min_value=1, max_value=100, step=1, key=mean_key)

        if family in SINGLE_PARAMETER_FAMILIES:
            # Poisson has only one free parameter -- showing a second control would imply it's
            # independently settable, which it isn't (Phase 1 feedback: "only give it the one
            # parameter"). Leave a blank spacer in its place so this column's rows still line up
            # with a two-parameter family's (e.g. deal size) in a side-by-side layout.
            _slider_placeholder(f"{key_prefix}_variance_placeholder")
        else:
            # Negative Binomial requires variance > mean, strictly -- shown to the user as a
            # Standard Deviation control instead of raw variance (Phase 1 feedback: "use Standard
            # Deviation instead of Variance"). 200 is still the fixed hard cap on variance (so the
            # SD cap is sqrt(200)); only the lower bound moves as the mean slider moves.
            min_variance = mean + 1
            max_variance = 200
            min_sd = float(np.sqrt(min_variance))
            max_sd = float(np.sqrt(max_variance))
            sd_key = f"{key_prefix}_sd"
            if sd_key not in st.session_state:
                default_variance = (
                    spec.variance
                    if (spec.variance and spec.mean is not None and spec.variance > spec.mean)
                    else mean * 2 + 1
                )
                default_sd = float(np.sqrt(np.clip(default_variance, min_variance, max_variance)))
                st.session_state[sd_key] = default_sd
            # If the mean slider was just dragged past the stored SD's implied variance (or the
            # fixed cap now sits below it), the current value is no longer valid -- snap it to
            # the new lowest legal value. Otherwise leave whatever value the user already chose
            # untouched (Phase 1 feedback: "stays the same unless it became invalid, in which
            # case update it to be the lowest legal value").
            if st.session_state[sd_key] ** 2 <= mean:
                st.session_state[sd_key] = min_sd
            st.session_state[sd_key] = float(np.clip(st.session_state[sd_key], min_sd, max_sd))
            sd = st.slider(
                "Standard deviation", min_value=min_sd, max_value=max_sd, step=0.1, key=sd_key
            )
            variance = sd**2

    spec_result = ParamSpec(family, mean=mean, variance=variance, low=low, high=high)
    try:
        resolved = spec_result.resolve()
        _render_preview(resolved)
    except DistributionConfigError as exc:
        st.error(str(exc))

    return spec_result


def deal_size_picker(spec: ParamSpec, key_prefix: str = "deal_size") -> ParamSpec:
    """Family + a second control (standard deviation, or low/high) picker for deal size, with the
    $0 floor enforced at generation time via rejection sampling
    (`distributions.sample_positive`) -- previewed here with the matching truncated-and-
    renormalized density (`lower_bound=0.0`).
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
            "Deal size has a hard $0 floor: Gamma/Lognormal are positive by construction, "
            "Normal enforces it by redrawing any value that lands at or below $0, and "
            "Continuous Uniform's own Low endpoint already starts at $50."
        ),
    )

    mean: float | None = None
    variance: float | None = None
    low: float | None = None
    high: float | None = None

    if family in UNIFORM_FAMILIES:
        # Hard $0 floor, structurally, same as the other families: both endpoints' sliders are
        # bounded $50-$10,000 (Phase 1 feedback: "For deal size, the lowest is $50, and the
        # highest is $10000").
        low_key = f"{key_prefix}_low"
        if low_key not in st.session_state:
            default_low = spec.low if spec.low is not None else 50.0
            st.session_state[low_key] = float(np.clip(default_low, 50.0, 9_950.0))
        # Low's own slider tops out one step below the overall $10,000 cap -- same reasoning as
        # lead arrival's Low/High: Streamlit's slider rejects a min_value == max_value, so High
        # always needs at least one step of headroom above Low.
        st.session_state[low_key] = float(np.clip(st.session_state[low_key], 50.0, 9_950.0))
        low = st.slider(
            "Low", min_value=50.0, max_value=9_950.0, step=50.0, format="$%.0f", key=low_key
        )

        true_min_high = low + 50.0  # the mathematically correct lower bound (may be $10,000 itself).
        slider_min_high = min(true_min_high, 9_950.0)  # keeps High's min_value < max_value=$10,000.
        high_key = f"{key_prefix}_high"
        if high_key not in st.session_state:
            default_high = spec.high if spec.high is not None else 10_000.0
            st.session_state[high_key] = float(np.clip(default_high, true_min_high, 10_000.0))
        if st.session_state[high_key] < true_min_high:
            st.session_state[high_key] = true_min_high
        st.session_state[high_key] = float(np.clip(st.session_state[high_key], slider_min_high, 10_000.0))
        high = st.slider(
            "High", min_value=slider_min_high, max_value=10_000.0, step=50.0, format="$%.0f", key=high_key
        )
    else:
        # Hard $0 floor, structurally: the mean slider itself never offers a value at or below
        # $0 -- for any of these families -- rather than allowing the selection and catching it
        # with an error afterward (Phase 1 feedback: "Don't let the user even select below
        # that").
        mean_key = f"{key_prefix}_mean"
        if mean_key not in st.session_state:
            default_mean = spec.mean if spec.mean is not None else 5_000.0
            st.session_state[mean_key] = float(np.clip(default_mean, 50.0, 10_000.0))
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

    spec_result = ParamSpec(family, mean=mean, variance=variance, low=low, high=high)
    try:
        validate_deal_size_spec(spec_result)
        resolved = spec_result.resolve()
        _render_preview(resolved, lower_bound=0.0)
    except (SimulationConfigError, DistributionConfigError) as exc:
        st.error(str(exc))

    return spec_result


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


def task_type_mix_picker(
    activity_type_mix: Mapping[TaskType, float], key_prefix: str = "activity_type_mix"
) -> dict[TaskType, float]:
    """Percentage-per-task-type sliders for how a rep's activities split across call / email /
    meeting / text -- same percentage-based, auto-normalized pattern as `industry_mix_picker`
    above (Phase 1 feedback: "Update the parameter configuration page with the relevant knobs
    for these new simulator actions").
    """
    st.subheader("Activity type mix")
    st.caption("Share of rep activities that are each task type.")

    task_types = list(TaskType)
    raw: dict[TaskType, float] = {}
    cols = st.columns(2)
    for i, task_type in enumerate(task_types):
        default_pct = activity_type_mix.get(task_type, 1.0 / len(task_types)) * 100.0
        pct_key = f"{key_prefix}_{task_type.value}_pct"
        if pct_key not in st.session_state:
            st.session_state[pct_key] = float(np.clip(round(default_pct, 1), 0.0, 100.0))
        with cols[i % 2]:
            raw[task_type] = st.slider(
                humanize(task_type.value),
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                format="%.0f%%",
                key=pct_key,
            )

    raw_total = sum(raw.values())
    if raw_total <= 0:
        st.error("At least one task type needs a nonzero weight.")
        return {task_type: 1.0 / len(task_types) for task_type in task_types}

    st.caption(f"Raw total: {raw_total:.0f}% -> normalized to 100%.")
    return {task_type: weight / raw_total for task_type, weight in raw.items()}


def rep_activity_picker(activity_prob: float, key_prefix: str = "rep_activity") -> float:
    """How likely an open lead gets a rep activity on any given simulated day (Phase 1 feedback:
    "Randomize rep activities on those leads"). Whether that activity converts the lead is no
    longer decided here -- see `lead_conversion_picker` below for the per-lead conversion model
    that replaced it.
    """
    st.subheader("Rep activity")
    return pct_slider(
        "Chance an open lead gets an activity today",
        activity_prob,
        key=f"{key_prefix}_activity_prob",
        min_pct=0.0,
        max_pct=100.0,
    )


def beta_prob_picker(label: str, spec: ParamSpec, key_prefix: str) -> ParamSpec:
    """Mean + standard-deviation picker locked to the Beta distribution, for a 0-1 probability
    the user thinks about as a percent -- a lead's base conversion probability, or its daily
    decay factor (Phase 1 feedback: "Update this so it's a beta distribution"). No family
    selector here: unlike lead arrival/deal size, these are always Beta, so a dropdown with one
    fixed option would just be clutter on an already busy page.
    """
    default_mean_pct = float(
        np.clip(round((spec.mean if spec.mean is not None else 0.2) * 100.0, 1), 1.0, 99.0)
    )
    mean_key = f"{key_prefix}_mean"
    if mean_key not in st.session_state:
        st.session_state[mean_key] = default_mean_pct
    st.session_state[mean_key] = float(np.clip(st.session_state[mean_key], 1.0, 99.0))
    mean_pct = st.slider(
        f"{label} (mean)", min_value=1.0, max_value=99.0, step=0.5, format="%.1f%%", key=mean_key
    )
    mean = mean_pct / 100.0

    # A Beta distribution's variance is capped by mean*(1-mean) -- stay a hair inside that so the
    # slider itself can never land on an invalid (mean, sd) combination.
    max_sd_pct = max(float(np.sqrt(mean * (1 - mean)) * 100.0 * 0.98), 0.2)
    sd_key = f"{key_prefix}_sd"
    default_sd_pct = float(np.sqrt(spec.variance)) * 100.0 if spec.variance else 5.0
    if sd_key not in st.session_state:
        st.session_state[sd_key] = float(np.clip(round(default_sd_pct, 1), 0.1, max_sd_pct))
    st.session_state[sd_key] = float(np.clip(st.session_state[sd_key], 0.1, max_sd_pct))
    sd_pct = st.slider(
        f"{label} (standard deviation)",
        min_value=0.1,
        max_value=max_sd_pct,
        step=0.1,
        format="%.1f%%",
        key=sd_key,
    )
    variance = (sd_pct / 100.0) ** 2

    spec_result = ParamSpec(Family.BETA, mean=mean, variance=variance)
    try:
        resolved = spec_result.resolve()
        _render_preview(resolved)
    except DistributionConfigError as exc:
        st.error(str(exc))

    return spec_result


def lead_conversion_picker(
    days_until_converted_mean: float,
    base_conversion_prob: ParamSpec,
    conversion_decay: ParamSpec,
    key_prefix: str = "lead_conversion",
) -> tuple[float, ParamSpec, ParamSpec]:
    """Per-lead conversion timing and probability -- assigned once when a Lead is generated, then
    evolves daily (decay) and can be nudged by rep activity (task_type_effects), rather than any
    single activity's outcome directly deciding conversion (Phase 1 feedback: "the chance of a
    lead being converted should not depend on the outcome of any particular activity").

    Tucked into a collapsed expander: these are the parameters people are least likely to touch
    day to day (Phase 1 feedback: "may want to simply hide some options later, like the decay
    value") -- collapsing the section is the simplest version of that for now.
    """
    with st.expander("Lead conversion timing & decay", expanded=False):
        mean_key = f"{key_prefix}_days_mean"
        if mean_key not in st.session_state:
            st.session_state[mean_key] = float(np.clip(days_until_converted_mean, 0.0, 60.0))
        st.session_state[mean_key] = float(np.clip(st.session_state[mean_key], 0.0, 60.0))
        days_mean = st.slider(
            "Average days until a lead would convert (0 = same-day)",
            min_value=0.0,
            max_value=60.0,
            step=0.5,
            key=mean_key,
            help=(
                "A lead that hasn't converted within this many days of being created is closed "
                "as Unqualified."
            ),
        )

        prob_col, decay_col = st.columns(2)
        with prob_col:
            base_conversion_prob = beta_prob_picker(
                "Base conversion probability",
                base_conversion_prob,
                key_prefix=f"{key_prefix}_base_prob",
            )
        with decay_col:
            conversion_decay = beta_prob_picker(
                "Daily decay if not converted", conversion_decay, key_prefix=f"{key_prefix}_decay"
            )

    return days_mean, base_conversion_prob, conversion_decay


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
