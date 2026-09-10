"""
Exploratory Data Analysis for NSL-KDD.

Produces a handful of meaningful plots (not dozens for the sake of it) into
notebooks/figures/, and prints the numeric findings that the README quotes.

Run:
    python src/eda.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
FIG_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def main():
    df = pd.read_csv(PROC_DIR / "train.csv")

    # 1. Class distribution (binary) — motivates the "don't trust accuracy" discussion
    plt.figure(figsize=(5, 4))
    df["label_binary"].map({0: "normal", 1: "attack"}).value_counts().plot(kind="bar", color=["#4C72B0", "#C44E52"])
    plt.title("Binary class distribution (train)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_distribution_binary.png", dpi=120)
    plt.close()

    # 2. Attack category distribution — motivates why R2L/U2R are hard (few samples)
    plt.figure(figsize=(6, 4))
    order = df["attack_category"].value_counts().index
    sns.countplot(data=df, x="attack_category", order=order, color="#55A868")
    plt.yscale("log")
    plt.title("Attack category distribution (log scale)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_distribution_category.png", dpi=120)
    plt.close()

    # 3. protocol_type vs label_binary — a genuinely useful categorical relationship
    plt.figure(figsize=(5, 4))
    ct = pd.crosstab(df["protocol_type"], df["label_binary"], normalize="index")
    ct.columns = ["normal", "attack"]
    ct.plot(kind="bar", stacked=True, color=["#4C72B0", "#C44E52"])
    plt.title("Attack rate by protocol_type")
    plt.ylabel("proportion")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "attack_rate_by_protocol.png", dpi=120)
    plt.close()

    # 4. Correlation heatmap of numeric features (subset — full 38 is unreadable)
    numeric_cols = [
        "duration", "src_bytes", "dst_bytes", "count", "srv_count",
        "serror_rate", "srv_serror_rate", "rerror_rate", "same_srv_rate",
        "diff_srv_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_serror_rate", "logged_in",
    ]
    plt.figure(figsize=(10, 8))
    corr = df[numeric_cols + ["label_binary"]].corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation heatmap (selected numeric features)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=120)
    plt.close()

    # 5. src_bytes distribution, log scale — heavy outliers are expected in network data
    plt.figure(figsize=(6, 4))
    plt.hist(df.loc[df.src_bytes > 0, "src_bytes"].apply(lambda x: __import__("math").log10(x)), bins=50, color="#4C72B0")
    plt.title("log10(src_bytes) distribution (src_bytes > 0)")
    plt.xlabel("log10(src_bytes)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "src_bytes_distribution.png", dpi=120)
    plt.close()

    # Numeric summaries the README/guide will quote directly
    print("=== Class balance (binary) ===")
    print(df["label_binary"].value_counts(normalize=True).round(3))

    print("\n=== Attack category counts ===")
    print(df["attack_category"].value_counts())

    print("\n=== Correlation of numeric features with label_binary (top 10 by |corr|) ===")
    full_numeric = df.select_dtypes(include="number").drop(columns=["label_binary"])
    corrs = full_numeric.apply(lambda col: col.corr(df["label_binary"])).abs().sort_values(ascending=False)
    print(corrs.head(10))

    print("\n=== Constant / near-constant columns (candidates to drop) ===")
    for col in full_numeric.columns:
        if df[col].nunique() <= 1:
            print(f"  {col}: constant, value={df[col].unique()}")

    print("\n=== protocol_type attack rate ===")
    print(df.groupby("protocol_type")["label_binary"].mean().round(3))


if __name__ == "__main__":
    main()
