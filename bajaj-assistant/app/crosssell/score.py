"""Loads model, scores a customer at runtime.

Never shown to customers directly — the propensity score, SHAP breakdown,
and fund tier this returns are consumed by the dev dashboard
(dashboard/app.py) and, later, the orchestrator's internal decision of
whether to bring up funds at all. SHAP explains what the model learned
from rule-derived weak labels (see app/crosssell/features.py), not an
independently verified truth about purchase likelihood.
"""

from functools import lru_cache

import joblib
import pandas as pd
import shap

from app.config import (
    CROSSSELL_FEATURE_COLUMNS_PATH,
    CROSSSELL_MODEL_PATH,
    CUSTOMER_CSV_PATH,
)
from app.crosssell.features import build_customer_features, to_model_matrix
from app.crosssell.rules import recommend_fund_tier

# Probability at/above which we recommend pitching a fund. Single named
# constant — imported by train.py (for the held-out confusion matrix) and
# by the dashboard, instead of being repeated as a magic number.
PITCH_THRESHOLD = 0.5

TOP_N_SHAP_FEATURES = 3


@lru_cache(maxsize=1)
def _load_artifacts():
    model = joblib.load(CROSSSELL_MODEL_PATH)
    feature_columns = joblib.load(CROSSSELL_FEATURE_COLUMNS_PATH)
    explainer = shap.TreeExplainer(model)
    return model, feature_columns, explainer


def score_customer(mobile_no, df: pd.DataFrame | None = None) -> dict | None:
    """Score one customer's fund cross-sell propensity.

    Returns None if the customer has no active loans — "not applicable",
    distinct from a low score of 0. `df` defaults to loading the full CSV
    but accepts a pre-loaded (or synthetic, for tests) frame so callers
    that already have the data in memory — the dashboard, tests — don't
    reload/re-filter the whole file per customer.
    """
    if df is None:
        df = pd.read_csv(CUSTOMER_CSV_PATH)

    mobile_no = int(mobile_no)  # MobileNo is int64 in the source data
    customer_rows = df[df["MobileNo"] == mobile_no]
    if customer_rows.empty:
        return None

    features_df = build_customer_features(customer_rows)
    if features_df.empty:
        return None

    row = features_df.iloc[0]

    model, feature_columns, explainer = _load_artifacts()
    X = to_model_matrix(features_df, columns=feature_columns)

    propensity = float(model.predict_proba(X)[0, 1])
    pitch_recommended = propensity >= PITCH_THRESHOLD

    fund_tier = recommend_fund_tier(int(row["age"]), row["employment_type"])

    shap_values = explainer.shap_values(X)  # shape: (1, n_features)
    contributions = shap_values[0]
    ranked = sorted(
        zip(feature_columns, X.iloc[0].tolist(), contributions.tolist()),
        key=lambda item: abs(item[2]),
        reverse=True,
    )
    top_features = [
        {"feature": name, "value": value, "shap_contribution": contribution}
        for name, value, contribution in ranked[:TOP_N_SHAP_FEATURES]
    ]

    return {
        "mobile_no": str(mobile_no),
        "propensity_score": round(propensity, 4),
        "pitch_recommended": bool(pitch_recommended),
        "fund_tier": fund_tier,
        "top_features": top_features,
    }
