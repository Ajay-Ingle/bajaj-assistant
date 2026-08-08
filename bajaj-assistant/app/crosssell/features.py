"""Feature engineering from loan data.

Builds one row of customer-level features per customer with at least one
ACTIVE loan (see granularity decision in Stage 2 spec), and derives a weak
supervision label from domain rules — there is no ground-truth "did this
customer buy a fund" label anywhere in the source data, so this label is a
stand-in encoding our best guess of who looks fund-ready, not an observed
outcome.

DATA NOTE: `DPD_30` / `DPD_90` in Customer_Data_.csv are NOT numeric day
counts despite the column names — they contain "YYYY-MM" strings marking
the month a 30+/90+ days-past-due bucket was first hit (confirmed by
inspection, see scripts/inspect_data.py output). A null value means that
bucket was never hit (clean). We therefore treat these as flags
(event-happened / never-happened) rather than taking a numeric max, which
is the closest faithful reading of "worst_dpd_30/90" given what the column
actually contains.
"""

import pandas as pd

# Feature matrix columns, in the fixed order used both when saving the
# trained model's expected input and when scoring a single customer at
# runtime (see app/crosssell/score.py).
FEATURE_COLUMNS = [
    "foir",
    "months_to_payoff",
    "dpd_clean",
    "app_score",
    "imputed_income",
    "loan_type_diversity",
    "total_loan_count",
    "age",
    "employment_type_self_employed",
]

# Weak-label rule threshold: FOIR (fixed obligations to income ratio) below
# this is considered healthy enough to safely pitch a fund. Defined once
# here rather than repeated as a magic number elsewhere.
FOIR_THRESHOLD = 0.40

_DATE_FORMAT = "%d-%m-%Y"
_AVG_DAYS_PER_MONTH = 30.44


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build one row of features per customer with >=1 active loan.

    `df` is the raw, loan-level customer CSV (or any subset of its rows,
    e.g. all rows for a single MobileNo — score.py uses this to score one
    customer without re-aggregating the whole population). Customers with
    zero active loans are excluded entirely (not scored, not a 0). Customers
    whose FOIR can't be computed (zero/null income) are also excluded
    rather than crashing or guessing an imputed value.
    """
    empty_result = pd.DataFrame(columns=[
        "mobile_no", "foir", "months_to_payoff", "dpd_clean", "app_score",
        "imputed_income", "loan_type_diversity", "total_loan_count", "age",
        "employment_type", "employment_type_self_employed", "gender",
        "total_emi", "total_outstanding_loan_amount",
        "worst_dpd_30", "worst_dpd_90",
    ])

    active = df[df["Loan_status"] == "Active"].copy()
    if active.empty:
        return empty_result

    disb_date = pd.to_datetime(active["Disbursement_date"], format=_DATE_FORMAT,
                                errors="coerce")
    unparseable = disb_date.isna()
    if unparseable.any():
        active = active[~unparseable].copy()
        disb_date = disb_date[~unparseable]
    if active.empty:
        return empty_result

    months_elapsed = (pd.Timestamp.now().normalize() - disb_date).dt.days / _AVG_DAYS_PER_MONTH
    active["_months_to_payoff"] = (active["Tenure"] - months_elapsed).clip(lower=0)

    # See module docstring DATA NOTE: presence of a value = the DPD bucket
    # was hit at some point on that loan; null = clean.
    active["_dpd30_flag"] = active["DPD_30"].notna().astype(int)
    active["_dpd90_flag"] = active["DPD_90"].notna().astype(int)

    per_customer = active.groupby("MobileNo").agg(
        total_emi=("EMI_amount", "sum"),
        total_outstanding_loan_amount=("Loan_amount", "sum"),
        worst_dpd_30=("_dpd30_flag", "max"),
        worst_dpd_90=("_dpd90_flag", "max"),
        avg_app_score=("App_score", "mean"),
        months_to_payoff=("_months_to_payoff", "min"),
        imputed_income=("Imputed_Income", "max"),
        age=("Age", "max"),
        employment_type=("Employment_type", "first"),
        gender=("Gender", "first"),
    ).reset_index()

    # loan_type_diversity / total_loan_count are relationship-depth proxies
    # computed over ALL of the customer's loans (active + closed), not just
    # the active ones used above.
    relationship = df.groupby("MobileNo").agg(
        loan_type_diversity=("Loan_type", "nunique"),
        total_loan_count=("LAN", "count"),
    ).reset_index()

    features = per_customer.merge(relationship, on="MobileNo", how="left")

    # FOIR: guard against division by zero/null income rather than crashing
    # or producing inf. Customers this happens to are excluded from the
    # returned frame (and therefore from training) rather than imputed.
    income = features["imputed_income"].where(features["imputed_income"] != 0)
    features["foir"] = features["total_emi"] / income

    before = len(features)
    features = features[features["foir"].notna()].copy()
    dropped = before - len(features)
    if dropped:
        print(f"NOTE: excluded {dropped} customer(s) — zero/null "
              f"Imputed_Income makes FOIR undefined for them.")
    if features.empty:
        return empty_result

    features["dpd_clean"] = (features["worst_dpd_30"] == 0) & (features["worst_dpd_90"] == 0)
    features["app_score"] = features["avg_app_score"]

    # Employment_type has exactly two observed values in this dataset
    # (Salaried, Self-employed) — a single binary column is the simplest
    # sensible encoding, equivalent to one-hot with drop_first=True, and
    # avoids a redundant second all-or-nothing column.
    features["employment_type_self_employed"] = (
        features["employment_type"].str.strip().str.lower() == "self-employed"
    ).astype(int)

    features = features.rename(columns={"MobileNo": "mobile_no"})

    return features[[
        "mobile_no", "foir", "months_to_payoff", "dpd_clean", "app_score",
        "imputed_income", "loan_type_diversity", "total_loan_count", "age",
        "employment_type", "employment_type_self_employed", "gender",
        "total_emi", "total_outstanding_loan_amount",
        "worst_dpd_30", "worst_dpd_90",
    ]].reset_index(drop=True)


def to_model_matrix(features_df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Select + order feature columns into the numeric matrix XGBoost expects.

    `columns` defaults to FEATURE_COLUMNS but accepts an explicit list (used
    by score.py, aligned against the column order saved at training time)
    so training and inference are guaranteed to line up even if
    FEATURE_COLUMNS changes in a future retrain.
    """
    columns = columns if columns is not None else FEATURE_COLUMNS
    return features_df[columns].astype(float)


def add_weak_labels(features_df: pd.DataFrame) -> pd.DataFrame:
    """Attach a weak-supervision `label` column via domain rules.

    label = 1 if FOIR is healthy, DPD history is clean, and the customer's
    app_score is at/above the population median — else 0. The median is
    computed dynamically from the customers being labeled, never hardcoded.

    This label is NOT a ground-truth purchase outcome (none exists in this
    dataset) — it is a hand-written proxy for "looks fund-ready," and the
    model trained on it (see train.py) should be read accordingly.
    """
    features_df = features_df.copy()
    app_score_median = features_df["app_score"].median()

    features_df["label"] = (
        (features_df["foir"] < FOIR_THRESHOLD)
        & (features_df["dpd_clean"])
        & (features_df["app_score"] >= app_score_median)
    ).astype(int)

    n = len(features_df)
    n_positive = int(features_df["label"].sum())
    positive_rate = (n_positive / n * 100) if n else 0.0
    print(f"weak label distribution: {n_positive}/{n} positive "
          f"({positive_rate:.1f}%); app_score median used as threshold: "
          f"{app_score_median:.2f}")
    if n and (positive_rate < 5 or positive_rate > 95):
        print(f"WARNING: weak label distribution is heavily imbalanced "
              f"({positive_rate:.1f}% positive) — inspect FOIR_THRESHOLD, "
              f"the DPD flags, and the app_score median before trusting a "
              f"model trained on this.")

    return features_df
