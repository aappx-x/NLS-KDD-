"""
Preprocessing pipeline for NSL-KDD.

Design decisions (explained in depth in the Interview Knowledge Guide):

1. Drop 'num_outbound_cmds' — constant (always 0), zero information.
2. Drop 'label' and 'attack_category' from the feature matrix — these ARE the
   answer, not features. Only 'label_binary' is the target.
3. Categorical columns (protocol_type, service, flag) -> One-Hot Encoding.
   Tree models don't need ordinal encoding, and one-hot avoids implying a
   false order between categories like 'http' and 'ftp'.
4. Numeric columns -> StandardScaler. Strictly speaking Random Forest does not
   need scaling (it's not distance-based), but Logistic Regression (our
   baseline) does. We scale once so the same processed matrix serves both
   models — this is standard practice and doesn't hurt the Random Forest.
5. Fit the encoder/scaler on TRAIN ONLY, then transform test with the fitted
   objects. Fitting on the full dataset (train+test combined) is a classic
   and subtle form of data leakage: it lets statistics from the test set
   (e.g. min/max, category presence) leak into how train data is represented.
6. The NSL-KDD test set contains a couple of 'service' categories not seen in
   train (e.g. certain rare services). handle_unknown='ignore' in the
   OneHotEncoder prevents a crash and is the correct production-realistic
   behavior: an unseen category at inference time gets an all-zero encoding.

Run:
    python src/preprocess.py
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from columns import CATEGORICAL_COLS

PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DROP_COLS = ["num_outbound_cmds"]
TARGET_COL = "label_binary"
NON_FEATURE_COLS = ["label", "attack_category", "label_binary"]


def build_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )


def main():
    train_df = pd.read_csv(PROC_DIR / "train.csv")
    test_df = pd.read_csv(PROC_DIR / "test.csv")

    for df in (train_df, test_df):
        df.drop(columns=DROP_COLS, inplace=True, errors="ignore")

    feature_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
    numeric_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]
    categorical_cols = CATEGORICAL_COLS

    X_train_raw = train_df[feature_cols]
    y_train = train_df[TARGET_COL].values
    X_test_raw = test_df[feature_cols]
    y_test = test_df[TARGET_COL].values

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train = preprocessor.fit_transform(X_train_raw)   # fit ONLY on train
    X_test = preprocessor.transform(X_test_raw)          # transform test with train's stats

    feature_names = (
        numeric_cols
        + list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols))
    )

    np.save(PROC_DIR / "X_train.npy", X_train)
    np.save(PROC_DIR / "X_test.npy", X_test)
    np.save(PROC_DIR / "y_train.npy", y_train)
    np.save(PROC_DIR / "y_test.npy", y_test)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.joblib")
    joblib.dump(feature_cols, MODELS_DIR / "raw_feature_cols.joblib")

    print("X_train:", X_train.shape, "X_test:", X_test.shape)
    print("Total encoded feature count:", len(feature_names))
    print("Class balance train:", np.bincount(y_train) / len(y_train))
    print("Class balance test:", np.bincount(y_test) / len(y_test))


if __name__ == "__main__":
    main()
