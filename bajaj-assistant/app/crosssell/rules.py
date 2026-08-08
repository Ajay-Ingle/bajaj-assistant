"""Fund-tier mapping (equity/debt/liquid) from score.

Separate from propensity: this answers "which fund category to suggest IF
we pitch," not "whether to pitch" (that's app/crosssell/score.py). Deciding
whether to pitch depends on the customer's financial health (FOIR,
delinquency, app score); deciding what to pitch is a simpler suitability
question, handled here with a plain rule instead of a model.
"""

# Age brackets are a starting heuristic, not a substitute for a real risk
# assessment — a production system would derive risk appetite from a
# questionnaire, not age alone.
_YOUNG_AGE_CEILING = 35
_MID_AGE_CEILING = 50

_TIER_ORDER = ["equity_growth", "hybrid_balanced", "debt_conservative"]


def recommend_fund_tier(age: int, employment_type: str) -> str:
    """Return one of "equity_growth", "hybrid_balanced", "debt_conservative".

    Primary signal is age (younger -> more growth-oriented, older -> more
    conservative). Self-employed customers are nudged one tier more
    conservative than a salaried customer of the same age, on the
    assumption that self-employed income tends to be less regular — a
    simplification, not a substitute for actually asking about income
    stability or risk tolerance.
    """
    if age < _YOUNG_AGE_CEILING:
        tier = "equity_growth"
    elif age <= _MID_AGE_CEILING:
        tier = "hybrid_balanced"
    else:
        tier = "debt_conservative"

    if employment_type and employment_type.strip().lower() == "self-employed":
        idx = min(_TIER_ORDER.index(tier) + 1, len(_TIER_ORDER) - 1)
        tier = _TIER_ORDER[idx]

    return tier
