"""
Trains two models on the preprocessed NSL-KDD data:

1. Logistic Regression — baseline. Simple, linear, fast, fully interpretable
   (each feature gets one coefficient). Establishes a floor: if Random Forest
   can't beat this by a meaningful margin, the added complexity isn't justified.

2. Random Forest — primary model. An ensemble of decision trees, each trained
   on a bootstrap sample with a random subset of features considered at each
   split. Chosen (not a neural network) because:
   - Network-flow data here is tabular, moderate-sized (126k rows, 121
     encoded features) — the regime where tree ensembles reliably match or
     beat deep learning, without needing GPU infra or thousands of epochs.
   - Handles non-linear feature interactions (e.g. combinations of
     same_srv_rate + serror_rate) that logistic regression can't capture
     with raw features.
   - Gives free, honest feature importances — critical for the
     interpretability requirement of this project.
   - Robust to unscaled/skewed numeric features (src_bytes has huge outliers)
     without needing transformation.

Run:
    python src/train.py
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_arrays():
    X_train = np.load(PROC_DIR / "X_train.npy")
    X_test = np.load(PROC_DIR / "X_test.npy")
    y_train = np.load(PROC_DIR / "y_train.npy")
    y_test = np.load(PROC_DIR / "y_test.npy")
    return X_train, X_test, y_train, y_test


def main():
    X_train, X_test, y_train, y_test = load_arrays()

    # --- Baseline: Logistic Regression ---
    # max_iter raised because the 121-dim one-hot encoded input needs more
    # iterations to converge than sklearn's default of 100.
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    lr_preds = log_reg.predict(X_test)
    lr_proba = log_reg.predict_proba(X_test)[:, 1]

    print("=" * 60)
    print("LOGISTIC REGRESSION (baseline)")
    print("=" * 60)
    print(classification_report(y_test, lr_preds, target_names=["normal", "attack"]))
    print("ROC-AUC:", round(roc_auc_score(y_test, lr_proba), 4))

    # --- Primary model: Random Forest (untuned, sensible defaults first) ---
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",  # mild correction, not oversampling — see guide
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print("RANDOM FOREST (untuned primary model)")
    print("=" * 60)
    print(classification_report(y_test, rf_preds, target_names=["normal", "attack"]))
    print("ROC-AUC:", round(roc_auc_score(y_test, rf_proba), 4))

    joblib.dump(log_reg, MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(rf, MODELS_DIR / "random_forest_untuned.joblib")


if __name__ == "__main__":
    main()
