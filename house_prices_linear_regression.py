"""
Ames Housing / Kaggle House Prices — linear regression baseline.

Pipeline: load train.csv -> drop leaky / sparse columns -> impute missing values
-> label-encode categoricals -> train/test split -> LinearRegression -> metrics
and an actual-vs-predicted scatter plot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# -----------------------------------------------------------------------------
# 1) Configuration
# -----------------------------------------------------------------------------
# Local Kaggle training file (same folder as this script).
DATA_PATH = Path(__file__).resolve().parent / "train.csv"
RANDOM_STATE = 42
# Columns with more than this fraction of missing values are dropped (too sparse
# to impute reliably; typical drops here include PoolQC, Alley, Fence, etc.).
MAX_NULL_FRACTION = 0.40
PLOT_PATH = Path(__file__).resolve().parent / "actual_vs_predicted.png"


def load_raw_data(csv_path: Path) -> pd.DataFrame:
    """Load the House Prices dataset from a local CSV file."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Expected dataset at {csv_path}")
    return pd.read_csv(csv_path)


def drop_high_null_and_id(df: pd.DataFrame, max_null_frac: float) -> pd.DataFrame:
    """
    Remove identifier and columns with a very high missing rate.

    - `Id` is not a predictive feature for price (just a row index from Kaggle).
    - Columns where most rows are NA carry little signal after crude imputation
      and can destabilize a simple linear model; dropping them is a common baseline.
    """
    df = df.copy()
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    n = len(df)
    null_frac = df.isna().sum() / n
    drop_cols = null_frac[null_frac > max_null_frac].index.tolist()
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values and label-encode every non-numeric column.

    - Numeric (integer/float) columns: fill missing values with the column median
      (robust to skewed distributions like area or year).
    - Object / string columns: fill missing values with the column mode (most
      frequent category), then apply sklearn LabelEncoder (0..K-1 integer codes).

    Note: Label encoding imposes an arbitrary order on nominal categories; it is
    requested here for simplicity. One-hot encoding is often preferred for linear models.
    """
    df = df.copy()

    # Split column types so we can impute and encode appropriately.
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # --- Numeric columns: median imputation ---
    for col in num_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # --- Categorical columns: mode imputation + label encoding ---
    for col in cat_cols:
        if df[col].isna().any():
            mode_series = df[col].mode(dropna=True)
            fill_value = mode_series.iloc[0] if len(mode_series) else "Missing"
            df[col] = df[col].fillna(fill_value)
        # Convert to string so LabelEncoder always sees consistent tokens.
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df


def build_design_matrix(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate target from features, drop sparse/irrelevant columns, then preprocess.

    Target `SalePrice` is what we predict (house sale price in dollars).
    """
    if "SalePrice" not in raw.columns:
        raise ValueError(
            "This script expects Kaggle House Prices train.csv with a 'SalePrice' column."
        )

    y = raw["SalePrice"]
    X_raw = raw.drop(columns=["SalePrice"])

    # Drop Id and very sparse columns before imputation/encoding.
    X_raw = drop_high_null_and_id(X_raw, MAX_NULL_FRACTION)

    X = preprocess_features(X_raw)
    return X, y


def main() -> None:
    # -------------------------------------------------------------------------
    # 2) Load data
    # -------------------------------------------------------------------------
    raw = load_raw_data(DATA_PATH)

    # -------------------------------------------------------------------------
    # 3) Preprocess (impute, encode, drop high-null / Id)
    # -------------------------------------------------------------------------
    X, y = build_design_matrix(raw)
    print(f"Feature matrix shape after preprocessing: {X.shape}")
    print(f"Features used (first 15): {list(X.columns[:15])} ...")

    # -------------------------------------------------------------------------
    # 4) Train / test split (80% train, 20% test)
    # -------------------------------------------------------------------------
    # Random split with fixed seed for reproducibility.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    # -------------------------------------------------------------------------
    # 5) Build and train Ordinary Least Squares Linear Regression
    # -------------------------------------------------------------------------
    # sklearn's LinearRegression fits coefficients that minimize MSE on the training set.
    model = LinearRegression()
    model.fit(X_train, y_train)

    # -------------------------------------------------------------------------
    # 6) Predict on the held-out test set and compute regression metrics
    # -------------------------------------------------------------------------
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)

    print("\n--- Test set metrics ---")
    print(f"MAE  (mean absolute error):     {mae:,.2f}")
    print(f"MSE  (mean squared error):      {mse:,.2f}")
    print(f"RMSE (root mean squared error): {rmse:,.2f}")
    print(f"R^2 (coefficient of determination): {r2:.4f}")

    # -------------------------------------------------------------------------
    # 7) Scatter plot: actual vs predicted sale prices
    # -------------------------------------------------------------------------
    # Points near the diagonal y=x would indicate perfect predictions.
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.5, edgecolors="none", s=20)
    lims = [
        min(y_test.min(), y_pred.min()),
        max(y_test.max(), y_pred.max()),
    ]
    plt.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction (y = x)")
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title("House Prices: actual vs predicted (test set)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"\nSaved scatter plot to: {PLOT_PATH.resolve()}")


if __name__ == "__main__":
    main()
