"""Display-text helpers for the Streamlit app (Phase 1 UI feedback): every dropdown option --
distribution family names, industry names, and anything else backed by a snake_case Enum value --
should read as clean Title Case in the UI, not as its raw Python identifier.
"""

from __future__ import annotations

# A few abbreviations that Title Case would otherwise mangle (`.capitalize()` on "saas" gives
# "Saas", not "SaaS"). Matched case-insensitively against each whitespace-split word.
_WORD_OVERRIDES = {
    "saas": "SaaS",
    "icp": "ICP",
}


def humanize(value: str) -> str:
    """Turn a snake_case enum value (e.g. "negative_binomial", "financial_services") into
    clean display text ("Negative Binomial", "Financial Services").
    """
    words = value.replace("_", " ").split()
    return " ".join(_WORD_OVERRIDES.get(word.lower(), word.capitalize()) for word in words)
