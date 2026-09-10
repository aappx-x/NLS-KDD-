"""
Loads the raw NSL-KDD text files, attaches proper column names, derives the
binary target (label_binary: normal vs attack) and the 5-class attack_category,
then writes clean CSVs to data/processed/.

Run once before anything else:
    python src/load_data.py
"""

import pandas as pd
from pathlib import Path
from columns import COLUMN_NAMES, ATTACK_CATEGORY_MAP

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


def load_split(filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    # 'difficulty' is a KDD-internal score of how hard a record is to classify.
    # It's not a real network feature and must NOT be used for training.
    df = df.drop(columns=["difficulty"])

    # NSL-KDD attack names have a trailing "." in some distributions; strip just in case.
    df["label"] = df["label"].str.strip().str.rstrip(".")

    # Binary target: what the classifier will actually predict in the demo.
    df["label_binary"] = (df["label"] != "normal").astype(int)

    # 5-class target: normal / dos / probe / r2l / u2r — used only for EDA and
    # to discuss "attack-type" in the interview guide. Unmapped/unseen attack
    # names (can happen in the test set) fall back to 'unknown_attack'.
    df["attack_category"] = df["label"].map(ATTACK_CATEGORY_MAP).fillna("unknown_attack")

    return df


def main():
    train_df = load_split("KDDTrain+.txt")
    test_df = load_split("KDDTest+.txt")

    train_df.to_csv(PROC_DIR / "train.csv", index=False)
    test_df.to_csv(PROC_DIR / "test.csv", index=False)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("\nTrain label_binary distribution:")
    print(train_df["label_binary"].value_counts(normalize=True))
    print("\nTrain attack_category distribution:")
    print(train_df["attack_category"].value_counts())


if __name__ == "__main__":
    main()
