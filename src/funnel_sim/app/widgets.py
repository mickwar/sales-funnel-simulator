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
    DAYS_UNTIL_CONVERTED_FAMILIES,
    DEAL_SIZE_FAMILIES,
    PROBABILITY_FAMILIES,
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


def _render_preview(
    resolved,
    key: str,
    lower_bound: float | None = None,
    fixed_range: tuple[float, float] | None = None,
) -> None:
    """Render the pdf/pmf preview chart right next to the parameter selection -- users called
    this out as the thing they want to see more of.

    `key` is required and must be unique per call site (it's just passed straight through to
    `st.plotly_chart`): two pickers can easily end up with pixel-identical preview data -- e.g.
    two Continuous-Uniform-on-[0,1] pickers both still on their matching defaults -- and without
    an explicit key, Streamlit's own auto-generated (type + params)-based element id collides in
    exactly that case.

    `fixed_range`, when given, both drives `preview_xy`'s own x-range (see its docstring) and
    pins the chart's displayed x-axis to exactly that range -- otherwise Plotly would still
    auto-fit the axis to whatever data came back, undoing the fixed range (Phase 1 feedback:
    "the plots should have fixed left and right end points").
    """
    try:
        x, y = preview_xy(resolved, lower_bound=lower_bound, fixed_range=fixed_range)
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
    if fixed_range is not None:
        fig.update_xaxes(range=list(fixed_range))
    st.plotly_chart(fig, use_container_width=True, key=key)
    for warning in resolved.warnings:
        st.info(warning)


def _range_slider(
    key: str,
    default_low: float,
    default_high: float,
    domain_lo: float,
    domain_hi: float,
    step: float,
    integer: bool,
    format: str | None = None,
) -> tuple[float, float]:
    """A single two-handle range slider for a Uniform family's (low, high) endpoints -- one
    widget instead of the earlier separate Low/High sliders (Phase 1 feedback: "update the UI
    for the uniform distributions to be a single slide bar with two points for the left and
    right endpoints"). Storing the pair as one `session_state` tuple also retires the old
    two-slider coupling problem entirely (Low's own `max_value` had to stay strictly below
    High's `min_value`, which broke down right at the domain's own edges, since Streamlit
    rejects a slider whose min_value == max_value) -- a single range slider's two handles simply
    can't cross, so no such edge case exists here.
    """
    cast = (lambda v: int(round(v))) if integer else float
    lo_bound, hi_bound = cast(domain_lo), cast(domain_hi)

    def _clamp_pair(lo: float, hi: float) -> tuple[float, float]:
        lo = cast(np.clip(lo, lo_bound, hi_bound))
        hi = cast(np.clip(hi, lo_bound, hi_bound))
        if hi <= lo:
            hi = min(lo + step, hi_bound)
            if hi <= lo:  # domain too narrow for even one step -- shouldn't happen in practice.
                lo = max(hi - step, lo_bound)
        return lo, hi

    if key not in st.session_state:
        st.session_state[key] = _clamp_pair(default_low, default_high)
    st.session_state[key] = _clamp_pair(*st.session_state[key])

    return st.slider(
        "Range (low, high)",
        min_value=lo_bound,
        max_value=hi_bound,
        step=step,
        format=format,
        key=key,
    )


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
    fixed_range: tuple[float, float] | None = None

    if family in UNIFORM_FAMILIES:
        # Lead count is a whole-number-per-day quantity, so the range slider is integer-stepped,
        # bounded 0-100 (Phase 1 feedback: "the lowest value can be 0 and the highest value is
        # 100").
        low, high = _range_slider(
            f"{key_prefix}_range",
            default_low=spec.low if spec.low is not None else 1,
            default_high=spec.high if spec.high is not None else 100,
            domain_lo=0,
            domain_hi=100,
            step=1,
            integer=True,
        )
        # The range slider is one row; the Mean+SD branch below is two -- a blank spacer keeps
        # this column's height matching a two-parameter family's in a side-by-side layout (see
        # _slider_placeholder's docstring).
        _slider_placeholder(f"{key_prefix}_variance_placeholder")
        # When Uniform is selected the preview always spans the full fixed domain, not just the
        # chosen low-high sub-range (Phase 1 feedback: "in lead arrival, the plot should always
        # go from x=0 to x=100").
        fixed_range = (0, 100)
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
        _render_preview(resolved, key=f"{key_prefix}_preview", fixed_range=fixed_range)
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
    fixed_range: tuple[float, float] | None = None

    if family in UNIFORM_FAMILIES:
        # Hard $0 floor, structurally, same as the other families: the range slider is bounded
        # $50-$10,000 (Phase 1 feedback: "For deal size, the lowest is $50, and the highest is
        # $10000").
        low, high = _range_slider(
            f"{key_prefix}_range",
            default_low=spec.low if spec.low is not None else 50.0,
            default_high=spec.high if spec.high is not None else 10_000.0,
            domain_lo=50.0,
            domain_hi=10_000.0,
            step=50.0,
            integer=False,
            format="$%.0f",
        )
        _slider_placeholder(f"{key_prefix}_variance_placeholder")
        # Phase 1 feedback: "for deal size, the plot should always go from x=$50 to x=$10000."
        fixed_range = (50.0, 10_000.0)
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
        _render_preview(
            resolved, key=f"{key_prefix}_preview", lower_bound=0.0, fixed_range=fixed_range
        )
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


def prob_dist_picker(label: str, spec: ParamSpec, key_prefix: str) -> ParamSpec:
    """Family + a second control (standard deviation, or a low-high range) picker for a 0-1
    probability the user thinks about as a percent -- a lead's base conversion probability, or
    its daily decay factor. Beta was the only option before; Continuous Uniform on [0, 1] was
    added per Phase 1 feedback ("Distributions should be selectable for Base conversion and
    Daily decay. Include Beta (already there) as well as Continuous Uniform on [0, 1]").

    `label` names the parameter once, as a caption above the whole picker -- it used to be baked
    into each slider's own label ("{label} (mean)", "{label} (standard deviation)"), which made
    it visually appear twice (Phase 1 feedback: "Base conversion probability now appears in two
    places").
    """
    st.caption(label)

    family_key = f"{key_prefix}_family"
    if family_key not in st.session_state:
        st.session_state[family_key] = spec.family
    family = st.selectbox(
        "Distribution",
        list(PROBABILITY_FAMILIES),
        format_func=lambda f: humanize(f.value),
        key=family_key,
    )

    mean: float | None = None
    variance: float | None = None
    low: float | None = None
    high: float | None = None

    if family in UNIFORM_FAMILIES:
        low_pct, high_pct = _range_slider(
            f"{key_prefix}_range_pct",
            default_low=(spec.low if spec.low is not None else 0.05) * 100.0,
            default_high=(spec.high if spec.high is not None else 0.95) * 100.0,
            domain_lo=0.0,
            domain_hi=100.0,
            step=0.5,
            integer=False,
            format="%.1f%%",
        )
        low, high = low_pct / 100.0, high_pct / 100.0
        _slider_placeholder(f"{key_prefix}_variance_placeholder")
    else:
        default_mean_pct = float(
            np.clip(round((spec.mean if spec.mean is not None else 0.2) * 100.0, 1), 1.0, 99.0)
        )
        mean_key = f"{key_prefix}_mean"
        if mean_key not in st.session_state:
            st.session_state[mean_key] = default_mean_pct
        st.session_state[mean_key] = float(np.clip(st.session_state[mean_key], 1.0, 99.0))
        mean_pct = st.slider(
            "Mean", min_value=1.0, max_value=99.0, step=0.5, format="%.1f%%", key=mean_key
        )
        mean = mean_pct / 100.0

        # A Beta distribution's variance is capped by mean*(1-mean) -- stay a hair inside that so
        # the slider itself can never land on an invalid (mean, sd) combination.
        max_sd_pct = max(float(np.sqrt(mean * (1 - mean)) * 100.0 * 0.98), 0.2)
        sd_key = f"{key_prefix}_sd"
        default_sd_pct = float(np.sqrt(spec.variance)) * 100.0 if spec.variance else 5.0
        if sd_key not in st.session_state:
            st.session_state[sd_key] = float(np.clip(round(default_sd_pct, 1), 0.1, max_sd_pct))
        st.session_state[sd_key] = float(np.clip(st.session_state[sd_key], 0.1, max_sd_pct))
        sd_pct = st.slider(
            "Standard deviation",
            min_value=0.1,
            max_value=max_sd_pct,
            step=0.1,
            format="%.1f%%",
            key=sd_key,
        )
        variance = (sd_pct / 100.0) ** 2

    spec_result = ParamSpec(family, mean=mean, variance=variance, low=low, high=high)
    try:
        resolved = spec_result.resolve()
        # Beta and Uniform(0, 1) are both fixed to the [0, 1] domain by construction, so the
        # preview always shows that full range (Phase 1 feedback: "when using beta or uniform(0,
        # 1), the plots should have a fixed x-range from 0 to 1").
        _render_preview(resolved, key=f"{key_prefix}_preview", fixed_range=(0.0, 1.0))
    except DistributionConfigError as exc:
        st.error(str(exc))

    return spec_result


def days_until_converted_picker(
    spec: ParamSpec, key_prefix: str = "lead_conversion_days"
) -> ParamSpec:
    """Family + a second control (standard deviation, or a low-high range) picker for how many
    days a lead has before being closed as Unqualified if it hasn't converted -- deliberately
    built the same way as `lead_arrival_picker` (Phase 1 feedback: "'Average days' should have a
    selected discrete distribution like lead arrivals ... The UI for this parameter should
    basically be the same as Lead arrivals"), just over [0, 60] days and offering Negative
    Binomial / Discrete Uniform rather than lead arrival's three count families (Phase 1
    feedback: "allow Negative Binomial and Discrete Uniform").
    """
    family_key = f"{key_prefix}_family"
    if family_key not in st.session_state:
        st.session_state[family_key] = spec.family
    family = st.selectbox(
        "Distribution",
        list(DAYS_UNTIL_CONVERTED_FAMILIES),
        format_func=lambda f: humanize(f.value),
        key=family_key,
        help=(
            "Negative Binomial lets you set the mean and standard deviation independently. "
            "Discrete Uniform makes every whole number of days in a range equally likely."
        ),
    )

    mean: float | None = None
    variance: float | None = None
    low: float | None = None
    high: float | None = None
    fixed_range: tuple[float, float] | None = None

    if family in UNIFORM_FAMILIES:
        low, high = _range_slider(
            f"{key_prefix}_range",
            default_low=spec.low if spec.low is not None else 0,
            default_high=spec.high if spec.high is not None else 60,
            domain_lo=0,
            domain_hi=60,
            step=1,
            integer=True,
        )
        _slider_placeholder(f"{key_prefix}_variance_placeholder")
        fixed_range = (0, 60)
    else:
        mean_key = f"{key_prefix}_mean"
        if mean_key not in st.session_state:
            default_mean = spec.mean if spec.mean is not None else 5
            st.session_state[mean_key] = int(np.clip(round(default_mean), 1, 60))
        st.session_state[mean_key] = int(np.clip(st.session_state[mean_key], 1, 60))
        mean = st.slider("Mean", min_value=1, max_value=60, step=1, key=mean_key)

        # Same Standard-Deviation-not-Variance treatment as lead arrival's Negative Binomial
        # branch -- 200 is still the fixed hard cap on variance.
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
        _render_preview(resolved, key=f"{key_prefix}_preview", fixed_range=fixed_range)
    except DistributionConfigError as exc:
        st.error(str(exc))

    return spec_result


def lead_conversion_picker(
    days_until_converted: ParamSpec,
    base_conversion_prob: ParamSpec,
    conversion_decay: ParamSpec,
    key_prefix: str = "lead_conversion",
) -> tuple[ParamSpec, ParamSpec, ParamSpec]:
    """Per-lead conversion timing and probability -- assigned once when a Lead is generated, then
    evolves daily (decay) and can be nudged by rep activity (task_type_effects), rather than any
    single activity's outcome directly deciding conversion (Phase 1 feedback: "the chance of a
    lead being converted should not depend on the outcome of any particular activity").

    Tucked into a collapsed expander: these are the parameters people are least likely to touch
    day to day (Phase 1 feedback: "may want to simply hide some options later, like the decay
    value") -- collapsing the section is the simplest version of that for now.
    """
    with st.expander("Lead conversion timing & decay", expanded=False):
        st.caption(
            "Average days until a lead would convert (0 = same-day). A lead that hasn't "
            "converted within its allotted days is closed as Unqualified."
        )
        days_until_converted = days_until_converted_picker(
            days_until_converted, key_prefix=f"{key_prefix}_days"
        )

        prob_col, decay_col = st.columns(2)
        with prob_col:
            base_conversion_prob = prob_dist_picker(
                "Base conversion probability",
                base_conversion_prob,
                key_prefix=f"{key_prefix}_base_prob",
            )
        with decay_col:
            conversion_decay = prob_dist_picker(
                "Daily decay if not converted", conversion_decay, key_prefix=f"{key_prefix}_decay"
            )

    return days_until_converted, base_conversion_prob, conversion_decay


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
