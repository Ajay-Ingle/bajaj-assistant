"""Trains + saves the propensity model (Stage 2).

IMPORTANT FRAMING: there is no ground-truth "did this customer buy a fund"
label in this dataset — it's loan data only. This is weak supervision:
domain rules (app/crosssell/features.py: add_weak_labels) encode a proxy
label, and this script trains an XGBoost classifier to reproduce that
proxy label smoothly across feature combinations the hand-written rule
didn't explicitly enumerate. The printed metrics below measure how well
the model reproduces the RULE on unseen customers — not real-world
purchase prediction accuracy, which this dataset cannot measure.

Run with: python -m app.crosssell.train
"""

import numpy as np
import pandas as pd
import shap
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from app.config import (
    CROSSSELL_FEATURE_COLUMNS_PATH,
    CROSSSELL_MODEL_PATH,
    CUSTOMER_CSV_PATH,
    MODELS_DIR,
)
from app.crosssell.features import FEATURE_COLUMNS, add_weak_labels, build_customer_features, to_model_matrix
from app.crosssell.score import PITCH_THRESHOLD


def main() -> None:
    print("=" * 60)
    print("LOADING DATA AND BUILDING FEATURES")
    print("=" * 60)
    df = pd.read_csv(CUSTOMER_CSV_PATH)
    features_df = build_customer_features(df)
    print(f"customers with an active loan and complete features: {len(features_df)}")

    print()
    print("=" * 60)
    print("WEAK LABEL DISTRIBUTION")
    print("=" * 60)
    labeled_df = add_weak_labels(features_df)

    X = to_model_matrix(labeled_df)
    y = labeled_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Small, sane defaults — this is a POC reproducing a hand-written rule,
    # not a competition model worth extensive hyperparameter search.
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= PITCH_THRESHOLD).astype(int)

    print()
    print("=" * 60)
    print("HELD-OUT METRICS")
    print("=" * 60)
    print(
        "NOTE: these numbers measure how well the model reproduces the "
        "weak-label rule on unseen customers, i.e. rule-reproduction "
        "accuracy. They are NOT a measurement of real-world fund purchase "
        "prediction accuracy — no purchase outcome exists in this dataset "
        "to validate against."
    )
    print(f"accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC:      {roc_auc_score(y_test, y_proba):.4f}")
    print(f"confusion matrix (threshold={PITCH_THRESHOLD}, rows=actual, "
          f"cols=predicted, order=[0, 1]):")
    print(confusion_matrix(y_test, y_pred))

    print()
    print("=" * 60)
    print("SHAP GLOBAL SUMMARY (mean |SHAP value| per feature, test set)")
    print("=" * 60)
    print(
        "NOTE: SHAP explains what the model learned from the rule-derived "
        "weak labels above, not an independently-verified truth about "
        "purchase likelihood. Use this to sanity-check the model is "
        "keying off sensible signals (FOIR, payoff proximity, DPD "
        "cleanliness), not something spurious."
    )
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    summary = pd.Series(mean_abs_shap, index=FEATURE_COLUMNS).sort_values(ascending=False)
    print(summary.to_string())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, CROSSSELL_MODEL_PATH)
    joblib.dump(FEATURE_COLUMNS, CROSSSELL_FEATURE_COLUMNS_PATH)
    print()
    print(f"saved model to {CROSSSELL_MODEL_PATH}")
    print(f"saved feature columns to {CROSSSELL_FEATURE_COLUMNS_PATH}")


if __name__ == "__main__":
    main()
