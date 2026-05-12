"""
Titanic survival binary classification using a small feed-forward neural network (Keras).

Steps: load CSV -> preprocess -> train/test split -> build model -> train with
validation split -> evaluate on held-out test set -> plot learning curves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras


# -----------------------------------------------------------------------------
# 1) Paths and reproducibility
# -----------------------------------------------------------------------------
# Resolve train.csv next to this script (Kaggle Titanic training file).
DATA_PATH = Path(__file__).resolve().parent / "train.csv"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def load_raw_data(csv_path: Path) -> pd.DataFrame:
    """Load the Titanic dataset from a local CSV file."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Expected dataset at {csv_path}")
    return pd.read_csv(csv_path)


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Clean and encode features for the neural network.

    - Drops identifiers / high-cardinality text columns that are not useful as tabular features.
    - Fills missing numeric values (Age, Fare) with the median (robust to outliers).
    - Fills missing Embarked with the mode (most common port).
    - Label-encodes Sex and Embarked to integer codes (suitable for dense layers).
    """
    df = df.copy()

    # Target: 1 = survived, 0 = did not survive
    target = df["Survived"]

    # Columns not used as model inputs (IDs / raw text / sparse cabin codes)
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin", "Survived"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # --- Missing values ---
    # Age has gaps; median is a simple stable imputation for this baseline model.
    df["Age"] = df["Age"].fillna(df["Age"].median())
    # Fare is usually complete in train.csv; fill defensively if any NaNs appear.
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    # Embarked: a few rows may be empty; use the most frequent value.
    if df["Embarked"].isna().any():
        mode_embarked = df["Embarked"].mode(dropna=True).iloc[0]
        df["Embarked"] = df["Embarked"].fillna(mode_embarked)

    # --- Categorical encoding ---
    # Sex: two categories -> 0/1
    le_sex = LabelEncoder()
    df["Sex"] = le_sex.fit_transform(df["Sex"].astype(str))

    # Embarked: S, C, Q -> integer labels
    le_embarked = LabelEncoder()
    df["Embarked"] = le_embarked.fit_transform(df["Embarked"].astype(str))

    return df, target


def build_model(input_dim: int) -> keras.Model:
    """
    Binary classifier: input -> two ReLU hidden layers -> sigmoid output.

    - ReLU hidden layers help the network learn non-linear patterns.
    - Sigmoid output gives a probability of class 1 (survived); threshold 0.5 at inference.
    """
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(64, activation="relu", name="hidden_1"),
            keras.layers.Dense(32, activation="relu", name="hidden_2"),
            keras.layers.Dense(1, activation="sigmoid", name="output"),
        ],
        name="titanic_mlp",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history: keras.callbacks.History, out_path: Path | None = None) -> None:
    """Plot training vs validation accuracy and loss across epochs."""
    hist = history.history
    epochs = range(1, len(hist["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, hist["accuracy"], label="Train accuracy")
    axes[0].plot(epochs, hist["val_accuracy"], label="Validation accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(epochs, hist["loss"], label="Train loss")
    axes[1].plot(epochs, hist["val_loss"], label="Validation loss")
    axes[1].set_title("Loss (binary crossentropy)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"\nSaved learning curves to: {out_path.resolve()}")
    plt.close(fig)


def main() -> None:
    # -------------------------------------------------------------------------
    # 2) Load and preprocess
    # -------------------------------------------------------------------------
    # Read Kaggle `train.csv` from disk, then build a numeric feature matrix X
    # and binary labels y (Survived). All cleaning/encoding lives in preprocess().
    raw = load_raw_data(DATA_PATH)
    X, y = preprocess(raw)

    feature_names = list(X.columns)
    n_features = X.shape[1]
    print(f"Features used ({n_features}): {feature_names}")

    # -------------------------------------------------------------------------
    # 3) Train / test split (80% train, 20% test), stratified by survival
    # -------------------------------------------------------------------------
    # Stratify keeps a similar proportion of survivors vs non-survivors in both
    # splits (helps when the target is imbalanced). Values are cast to float32
    # because Keras expects floating-point tensors for Dense layers.
    X_train, X_test, y_train, y_test = train_test_split(
        X.values.astype("float32"),
        y.values.astype("float32"),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # -------------------------------------------------------------------------
    # 4) Build the neural network
    # -------------------------------------------------------------------------
    # Input size matches the number of engineered columns (7 for this pipeline).
    model = build_model(input_dim=n_features)
    model.summary()

    # -------------------------------------------------------------------------
    # 5) Train with a validation slice so we get val metrics each epoch
    # -------------------------------------------------------------------------
    # Important distinction:
    # - X_train / y_train = 80% of the full data (the "training pool").
    # - validation_split=0.2 carves 20% of that pool for validation metrics
    #   printed every epoch (val_accuracy, val_loss). shuffle=True avoids order
    #   bias before that carve. This is NOT the same as the 20% test split above.
    # - X_test / y_test stay untouched until step 6 (true generalization check).
    # verbose=1 prints one line per epoch with train + validation metrics.
    history = model.fit(
        X_train,
        y_train,
        epochs=60,
        batch_size=32,
        validation_split=0.2,
        shuffle=True,
        verbose=1,
    )

    # -------------------------------------------------------------------------
    # 6) Evaluate on the held-out test set (never seen during training)
    # -------------------------------------------------------------------------
    # Sigmoid outputs a probability; 0.5 is the usual decision threshold for
    # class 1 (survived). Confusion matrix + classification report summarize
    # errors beyond a single accuracy number.
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    test_accuracy = accuracy_score(y_test.astype(int), y_pred)
    print(f"\nTest accuracy: {test_accuracy:.4f}")

    cm = confusion_matrix(y_test.astype(int), y_pred)
    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(cm)

    print("\nClassification report:")
    print(classification_report(y_test.astype(int), y_pred, digits=4))

    # -------------------------------------------------------------------------
    # 7) Plot learning curves (train vs validation)
    # -------------------------------------------------------------------------
    # Uses the History object returned by fit(); curves help spot overfitting
    # (train improving while validation stalls or worsens). Figure is saved to
    # disk so you can open it even when no GUI display is attached.
    plot_history(history, out_path=Path(__file__).resolve().parent / "training_curves.png")


if __name__ == "__main__":
    main()
