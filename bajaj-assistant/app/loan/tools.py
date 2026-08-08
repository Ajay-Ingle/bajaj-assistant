"""Deterministic lookup functions (Stage 1), wired up for Stage 4.

`get_loan_details` is the one function the LLM ever calls to answer loan
questions -- it returns exact values computed here in Python, never
numbers the model estimated itself. See app/orchestrator/llm_client.py.

DATA NOTE (carried over from Stage 2): `DPD_30` / `DPD_90` in the source
CSV are "YYYY-MM" strings marking the month a 30+/90+ DPD bucket was first
hit, not numeric day counts. A null value means that bucket was never hit.
`repayment_status` below treats presence/absence the same way Stage 2's
`dpd_clean` did: presence = "nonzero" (overdue at that bucket), null =
"zero" (clean).
"""

import pandas as pd

from app.loan.data_loader import load_customer_data

_DATE_FORMAT = "%d-%m-%Y"
_AVG_DAYS_PER_MONTH = 30.44


def list_loan_accounts(mobile_no: str) -> list[dict]:
    """Every loan (active + closed) for a mobile number.

    Returns an empty list if the mobile number isn't found or isn't a
    valid number -- the caller decides how to handle that (e.g. a 404 at
    the API layer), this function never raises for a bad/unknown number.
    """
    df = load_customer_data()
    try:
        mobile_no_int = int(mobile_no)
    except (TypeError, ValueError):
        return []

    rows = df[df["MobileNo"] == mobile_no_int]
    if rows.empty:
        return []

    return [
        {
            "lan": str(row["LAN"]),
            "loan_type": str(row["Loan_type"]),
            "loan_status": str(row["Loan_status"]),
            "disbursement_date": str(row["Disbursement_date"]),
        }
        for _, row in rows.iterrows()
    ]


def get_loan_details(lan: str) -> dict | None:
    """The single tool exposed to the LLM. Returns None if the LAN doesn't exist."""
    df = load_customer_data()
    rows = df[df["LAN"] == lan]
    if rows.empty:
        return None
    row = rows.iloc[0]

    tenure_months = int(row["Tenure"])

    disb_date = pd.to_datetime(row["Disbursement_date"], format=_DATE_FORMAT, errors="coerce")
    if pd.isna(disb_date):
        months_elapsed = 0
    else:
        months_elapsed = max(
            int((pd.Timestamp.now().normalize() - disb_date).days / _AVG_DAYS_PER_MONTH), 0
        )
    months_remaining = max(tenure_months - months_elapsed, 0)

    # Approximation: no outstanding-principal column exists in this data.
    # Uses straight-line amortization (loan_amount * months_remaining /
    # tenure_months) as a POC simplification. Production would pull the
    # actual outstanding balance from the loan servicing system.
    if tenure_months > 0:
        approx_outstanding_balance = int(
            round(row["Loan_amount"] * months_remaining / tenure_months)
        )
    else:
        approx_outstanding_balance = 0

    loan_status = str(row["Loan_status"])
    dpd_30 = row.get("DPD_30")
    dpd_90 = row.get("DPD_90")

    if loan_status == "Closed":
        repayment_status = "closed_fully_repaid"
    elif pd.notna(dpd_90):
        repayment_status = "overdue_90"
    elif pd.notna(dpd_30):
        repayment_status = "overdue_30"
    else:
        repayment_status = "current"
    # NOTE: Loan_status also has a rare "NPA" value (7/10000 rows) that
    # this rule doesn't special-case -- an NPA loan with no DPD marker set
    # would fall through to "current", which understates it. Flagging
    # rather than silently handling, since the spec's repayment_status
    # rule only distinguishes Closed vs. DPD-based status.

    return {
        "lan": str(row["LAN"]),
        "loan_type": str(row["Loan_type"]),
        "loan_status": loan_status,
        "emi_amount": int(row["EMI_amount"]),
        "tenure_months": tenure_months,
        "months_elapsed": months_elapsed,
        "months_remaining": months_remaining,
        "approx_outstanding_balance": approx_outstanding_balance,
        "repayment_status": repayment_status,
    }
