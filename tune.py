"""
Hyperparameter tuning for the Random Forest model using RandomizedSearchCV.

Why RandomizedSearchCV over GridSearchCV:
  GridSearchCV tries every combination of the given parameter grid — with 5
  parameters and even 3-4 values each, that's hundreds of model fits. On 126k
  rows x 121 features that's slow and mostly wasted (many combinations are
  redundant). RandomizedSearchCV samples a fixed number of random
  combinations from the same grid, which in practice finds a near-equally
  good configuration in a fraction of the time. This is the standard
  trade-off, not a shortcut taken to hide understanding.

Why tuning uses ONLY the training set (via cross-validation), never the test set:
  5-fold CV splits X_train into 5 folds; for each candidate hyperparameter
  set, 4 folds train the model and the 5th (held out) fold scores it, rotated
  5 times. The test set (X_test/y_test) is never touched during this process.
  If we tuned against the test set instead, we would be indirectly fitting
  hyperparameters to that specific test data — the reported test score would
  then be optimistic and would not reflect how the model behaves on truly new
  data. This is the same data-leakage principle as fitting the scaler on test
  data, applied to model selection instead of feature transformation.

Why F1 (not accuracy) is the scoring metric for the search:
  The dataset is not wildly imbalanced (53/47) but F1 balances precision and
  recall directly, which is the trade-off that matters most for an IDS (see
  Interview Knowledge Guide, Evaluation section). Optimizing for raw accuracy
  during tuning could quietly favor configurations that sacrifice recall.

Run:
    python src/tune.py
"""

import numpy as np
import joblib
from pathlib import Path
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main():
    X_train = np.load(PROC_DIR / "X_train.npy")
    X_test = np.load(PROC_DIR / "X_test.npy")
    y_train = np.load(PROC_DIR / "y_train.npy")
    y_test = np.load(PROC_DIR / "y_test.npy")

    # Parameter distributions. Each has a direct, explainable effect:
    #   n_estimators      -> number of trees. More trees = more stable
    #                        predictions, diminishing returns, higher cost.
    #   max_depth         -> how deep each tree can grow. Shallow trees
    #                        underfit; unlimited depth can overfit noisy rows.
    #   min_samples_split -> minimum samples required to split a node. Higher
    #                        values force simpler trees (regularization).
    #   min_samples_leaf  -> minimum samples in a leaf. Higher values smooth
    #                        out predictions and reduce overfitting to
    #                        individual outlier rows.
    #   max_features      -> number of features considered at each split.
    #                        Lower values decorrelate trees (more diversity
    #                        across the ensemble) at some cost to per-tree
    #                        accuracy.
    param_distributions = {
        "n_estimators": randint(100, 300),
        "max_depth": [10, 20, 30, None],
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2", None],
    }

    base_rf = RandomForestClassifier(
        random_state=42, n_jobs=-1, class_weight="balanced_subsample"
    )

    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_distributions,
        n_iter=10,          # 10 random configs x 3 folds = 30 fits (keeps runtime student-realistic)
        scoring="f1",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)  # test set not passed in anywhere here

    print("Best params found:", search.best_params_)
    print("Best CV F1 score:", round(search.best_score_, 4))

    best_rf = search.best_estimator_
    preds = best_rf.predict(X_test)
    proba = best_rf.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print("RANDOM FOREST (tuned) — held-out TEST SET performance")
    print("=" * 60)
    print(classification_report(y_test, preds, target_names=["normal", "attack"]))
    print("ROC-AUC:", round(roc_auc_score(y_test, proba), 4))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    joblib.dump(best_rf, MODELS_DIR / "random_forest_tuned.joblib")
    joblib.dump(search.best_params_, MODELS_DIR / "best_params.joblib")


if __name__ == "__main__":
    main()
