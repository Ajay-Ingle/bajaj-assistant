"""Fund tier -> pitchable fund mapping.

Closes the loop between app.crosssell.rules.recommend_fund_tier()'s output
(a tier label) and an actual, nameable fund from the factsheet's canonical
list (app.funds.ingest.CANONICAL_FUNDS). Deciding "whether to pitch" and
"which tier" happens in app/crosssell/; this module only answers "given a
tier, which specific fund do we name."
"""

TIER_TO_FUNDS = {
    "equity_growth": [
        "Bajaj Finserv Flexi Cap Fund",
        "Bajaj Finserv Multi Cap Fund",
    ],
    "hybrid_balanced": [
        "Bajaj Finserv Balanced Advantage Fund",
        "Bajaj Finserv Multi Asset Allocation Fund",
    ],
    "debt_conservative": [
        "Bajaj Finserv Low Duration Fund",
        "Bajaj Finserv Banking and PSU Fund",
    ],
}


def get_pitchable_fund(tier: str) -> str:
    """Return one concrete fund name to pitch for a given tier.

    Simple, deterministic choice for the POC — always the first fund
    listed for that tier. A production version might rotate or personalize
    which fund within the tier gets surfaced (e.g. by past engagement or
    fund-level capacity), rather than always naming the same one.
    """
    return TIER_TO_FUNDS[tier][0]
