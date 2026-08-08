"""Tests for app.crosssell (features, weak labels, and runtime scoring)."""

import pandas as pd

from app.crosssell.features import add_weak_labels, build_customer_features
from app.crosssell.score import score_customer


def _make_loan_row(**overrides):
    row = {
        "MobileNo": 9000000001,
        "Name": "Test Customer",
        "Acquiring_channel": "Branch Referral",
        "Conversion_attribution": "App",
        "Type": "New",
        "LAN": "LAN0000000001",
        "Loan_amount": 100000,
        "Interest": 0.12,
        "Tenure": 24,
        "Principal_amount": 100000,
        "Loan_type": "Personal loan",
        "DPD_30": None,
        "DPD_90": None,
        "App_score": 8.0,
        "Imputed_Income": 50000,
        "PIncode": 400001,
        "Loan_status": "Active",
        "Disbursement_date": "01-01-2025",
        "EMI_amount": 5000,
        "Age": 30,
        "Gender": "Female",
        "Employment_type": "Salaried",
    }
    row.update(overrides)
    return row


def test_foir_handles_zero_income_without_crashing():
    df = pd.DataFrame([
        _make_loan_row(MobileNo=1, Imputed_Income=0),
        _make_loan_row(MobileNo=2, LAN="LAN0000000002"),
    ])

    features = build_customer_features(df)

    # The zero-income customer is excluded (FOIR undefined), not crashed on
    # and not silently given a guessed value.
    assert 1 not in features["mobile_no"].values
    assert 2 in features["mobile_no"].values
    assert features["foir"].notna().all()


def test_score_customer_with_no_active_loans_returns_none():
    df = pd.DataFrame([
        _make_loan_row(MobileNo=3, Loan_status="Closed"),
    ])

    result = score_customer(3, df=df)

    assert result is None


def test_weak_label_rule_produces_both_classes():
    features = pd.DataFrame([
        # Healthy customer: low FOIR, clean DPD, high app_score -> label 1
        {"mobile_no": 1, "foir": 0.10, "dpd_clean": True, "app_score": 9.0},
        {"mobile_no": 2, "foir": 0.15, "dpd_clean": True, "app_score": 8.5},
        # Unhealthy customer: high FOIR -> label 0
        {"mobile_no": 3, "foir": 0.80, "dpd_clean": True, "app_score": 9.0},
        {"mobile_no": 4, "foir": 0.90, "dpd_clean": False, "app_score": 4.0},
    ])

    labeled = add_weak_labels(features)

    assert set(labeled["label"].unique()) == {0, 1}
