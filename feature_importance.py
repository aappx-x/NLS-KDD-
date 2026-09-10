"""
Feature importance for the trained Random Forest.

Two methods, both interpretable and standard (no SHAP — not needed here):

1. Built-in RF importance (Gini importance / mean decrease in impurity):
   for every split in every tree that uses a given feature, measure how much
   that split reduced impurity (Gini index), then average across all trees.
   Fast (computed as a byproduct of training) but has a known bias: it can
   inflate the importance of high-cardinality features (like one-hot encoded
   'service', which has ~70 columns).

2. Permutation importance: for a fitted model, shuffle one feature column at
   a time in the test set and measure how much performance (here, F1) drops.
   A feature whose shuffling barely changes performance wasn't being relied
   on; a large drop means the model genuinely needs that feature. This
   method is slower (it re-scores the model once per feature) but is not
   biased by cardinality, which is why we report both and note where they
   disagree.

Run (after train.py or tune.py has produced a model):
    python src/feature_importance.py
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
FIG_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "figures"


def get_model_path():
    tuned = MODELS_DIR / "random_forest_tuned.joblib"
    untuned = MODELS_DIR / "random_forest_untuned.joblib"
    return tuned if tuned.exists() else untuned


def main():
    model_path = get_model_path()
    print("Using model:", model_path.name)
    rf = joblib.load(model_path)
    feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")

    X_test = np.load(PROC_DIR / "X_test.npy")
    y_test = np.load(PROC_DIR / "y_test.npy")

    # --- 1. Built-in importance ---
    builtin = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\nTop 15 features by built-in (Gini) importance:")
    print(builtin.head(15))

    plt.figure(figsize=(8, 6))
    builtin.head(15)[::-1].plot(kind="barh", color="#4C72B0")
    plt.title("Top 15 features — Random Forest built-in importance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance_builtin.png", dpi=120)
    plt.close()

    # --- 2. Permutation importance (subsample test set for speed on a student machine) ---
    rng = np.random.RandomState(42)
    sample_size = min(5000, X_test.shape[0])
    idx = rng.choice(X_test.shape[0], size=sample_size, replace=False)
    X_sample, y_sample = X_test[idx], y_test[idx]

    perm = permutation_importance(
        rf, X_sample, y_sample, n_repeats=5, random_state=42, scoring="f1", n_jobs=1
    )
    perm_series = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)
    print("\nTop 15 features by permutation importance (F1 drop):")
    print(perm_series.head(15))

    plt.figure(figsize=(8, 6))
    perm_series.head(15)[::-1].plot(kind="barh", color="#C44E52")
    plt.title("Top 15 features — Permutation importance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_importance_permutation.png", dpi=120)
    plt.close()

    joblib.dump(builtin, MODELS_DIR / "importance_builtin.joblib")
    joblib.dump(perm_series, MODELS_DIR / "importance_permutation.joblib")


if __name__ == "__main__":
    main()
