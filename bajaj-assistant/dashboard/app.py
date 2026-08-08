"""Developer dashboard for inspecting cross-sell propensity scoring.

Internal tool only — never shown to end customers. Lets a developer pick a
customer and see their raw active loan data, computed features, propensity
score, SHAP-driven explanation, and recommended fund tier, in order to
sanity-check the model (app/crosssell/train.py, score.py), not to serve
any customer-facing purpose. Kept in its own top-level folder, separate
from frontend/, so that boundary is structural, not just a comment.

Run with: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Allow `import app.*` when Streamlit runs this file directly — it isn't
# executed as part of the `app` package, so the project root needs to be
# on sys.path first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CROSSSELL_MODEL_PATH, CUSTOMER_CSV_PATH  # noqa: E402
from app.crosssell.features import build_customer_features  # noqa: E402
from app.crosssell.score import score_customer  # noqa: E402

st.set_page_config(page_title="Cross-sell model dashboard", layout="wide")
st.title("Cross-sell propensity — developer dashboard")
st.caption(
    "Propensity is estimated from weak, rule-derived labels — this "
    "dashboard is for internal model verification, not shown to customers."
)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(CUSTOMER_CSV_PATH)


@st.cache_data
def load_all_features(df: pd.DataFrame) -> pd.DataFrame:
    return build_customer_features(df)


df = load_data()

if not CROSSSELL_MODEL_PATH.exists():
    st.error(
        f"No trained model found at {CROSSSELL_MODEL_PATH}. Run "
        f"`python -m app.crosssell.train` first, then reload this page."
    )
    st.stop()

all_features = load_all_features(df)
if all_features.empty:
    st.error("No customers currently have complete, active-loan feature data to score.")
    st.stop()

scoreable_mobiles = sorted(all_features["mobile_no"].unique().tolist())
selected_mobile = st.selectbox(
    "Customer (MobileNo) — active-loan holders that can be scored",
    options=scoreable_mobiles,
)

customer_rows = df[df["MobileNo"] == selected_mobile]
active_rows = customer_rows[customer_rows["Loan_status"] == "Active"]

st.subheader("Raw active loan row(s)")
st.dataframe(active_rows, use_container_width=True)

st.subheader("Computed features")
customer_features = all_features[all_features["mobile_no"] == selected_mobile]
st.dataframe(customer_features, use_container_width=True)

result = score_customer(selected_mobile, df=df)

if result is None:
    # Shouldn't happen given scoreable_mobiles is derived from the same
    # feature builder, but score_customer is the source of truth — trust
    # its answer, not the dropdown's assumption.
    st.warning("This customer could not be scored.")
else:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Propensity score", f"{result['propensity_score']:.2%}")
        st.write(
            "Pitch recommended: "
            + ("✅ Yes" if result["pitch_recommended"] else "❌ No")
        )
    with col2:
        st.metric("Recommended fund tier", result["fund_tier"])

    st.subheader("Top SHAP feature contributions")
    st.caption(
        "Explains what the model learned from the rule-derived weak "
        "labels above — not an independently-verified truth about "
        "purchase likelihood. For internal verification only."
    )
    top = result["top_features"]
    names = [f"{t['feature']} = {t['value']:.3g}" for t in top][::-1]
    contributions = [t["shap_contribution"] for t in top][::-1]
    colors = ["#d62728" if c < 0 else "#2ca02c" for c in contributions]

    fig, ax = plt.subplots()
    ax.barh(names, contributions, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP contribution (pushes propensity up / down)")
    st.pyplot(fig)

st.divider()
st.subheader("Orchestrator observability (coming in Stage 5)")
# Placeholder slot only — token usage / latency views land here once
# app/observability/logger.py exists. Not built yet.
