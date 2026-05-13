"""
Titanic survival — binary classification with scikit-learn Logistic Regression.

Pipeline: load train.csv -> preprocess (impute, label-encode, drop columns) ->
80/20 split -> fit logistic regression -> test metrics + confusion matrix heatmap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# -----------------------------------------------------------------------------
# 1) Paths and reproducibility
# -----------------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parent / "train.csv"
RANDOM_STATE = 42
PLOT_PATH = Path(__file__).resolve().parent / "confusion_matrix_heatmap.png"
np.random.seed(RANDOM_STATE)


def load_raw_data(csv_path: Path) -> pd.DataFrame:
    """Load the Titanic training CSV from disk (Kaggle `train.csv`)."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Expected dataset at {csv_path}")
    return pd.read_csv(csv_path)


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and binary target y (Survived).

    Missing values (per requirements):
    - Age: filled with the median age (stable for skewed gaps).
    - Embarked: filled with the mode (most common port).

    Other numeric columns (e.g. Fare) are filled with the median if any NaNs
    remain so scikit-learn receives a fully numeric matrix without NaNs.

    Categorical encoding:
    - Sex and Embarked are label-encoded (integer codes 0..K-1).

    Dropped columns (not used as model inputs):
    - PassengerId, Name, Ticket, Cabin (identifiers or high-cardinality text).
    """
    df = df.copy()

    if "Survived" not in df.columns:
        raise ValueError("Expected a 'Survived' column (Kaggle Titanic train.csv).")

    y = df["Survived"]

    # Drop columns that should not enter the model as raw text or IDs.
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin", "Survived"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # --- Missing values (required + minimal extra for numeric completeness) ---
    X["Age"] = X["Age"].fillna(X["Age"].median())
    if X["Embarked"].isna().any():
        mode_embarked = X["Embarked"].mode(dropna=True).iloc[0]
        X["Embarked"] = X["Embarked"].fillna(mode_embarked)
    # Fare is usually complete; impute median if any NaNs so the model never sees NaN.
    X["Fare"] = X["Fare"].fillna(X["Fare"].median())

    # --- Label encoding for Sex and Embarked ---
    le_sex = LabelEncoder()
    X["Sex"] = le_sex.fit_transform(X["Sex"].astype(str))

    le_embarked = LabelEncoder()
    X["Embarked"] = le_embarked.fit_transform(X["Embarked"].astype(str))

    return X, y


def plot_confusion_matrix_heatmap(cm: np.ndarray, out_path: Path) -> None:
    """
    Draw the confusion matrix as a labeled heatmap (counts in each cell).

    Rows correspond to true labels, columns to predicted labels — same layout
    as sklearn.metrics.confusion_matrix by default.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Titanic survival — confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved confusion matrix heatmap to: {out_path.resolve()}")


def main() -> None:
    # -------------------------------------------------------------------------
    # 2) Load dataset
    # -------------------------------------------------------------------------
    raw = load_raw_data(DATA_PATH)

    # -------------------------------------------------------------------------
    # 3) Preprocess features and target
    # -------------------------------------------------------------------------
    X, y = preprocess(raw)
    print(f"Features used ({X.shape[1]}): {list(X.columns)}")

    # -------------------------------------------------------------------------
    # 4) Train / test split (80% train, 20% test)
    # -------------------------------------------------------------------------
    # Stratify on Survived so both splits keep similar class proportions.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # -------------------------------------------------------------------------
    # 5) Logistic Regression — train on the training split
    # -------------------------------------------------------------------------
    # max_iter increased so the solver reliably converges on this small dataset.
    model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # -------------------------------------------------------------------------
    # 6) Evaluate on the held-out test set
    # -------------------------------------------------------------------------
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nTest accuracy: {acc:.4f}")
    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    # -------------------------------------------------------------------------
    # 7) Confusion matrix heatmap
    # -------------------------------------------------------------------------
    plot_confusion_matrix_heatmap(cm, PLOT_PATH)


if __name__ == "__main__":
    main()
