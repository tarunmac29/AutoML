"""
preprocessing.py
Handles data cleaning, missing value imputation, encoding, and splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df: pd.DataFrame, target_col: str):
    """
    Full preprocessing pipeline:
    - Remove duplicates
    - Fill missing values (mean for numeric, mode for categorical)
    - One-hot encode categorical features
    - Separate features (X) and target (y)
    - Encode target if classification

    Args:
        df: Raw pandas DataFrame
        target_col: Name of target column

    Returns:
        X (DataFrame), y (Series), label_encoder (or None), feature_names (list)
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    df = df.copy()

    # --- Remove duplicates ---
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    removed = before - after

    # --- Separate features and target ---
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # --- Fill missing values ---
    for col in X.columns:
        if X[col].dtype in [np.float64, np.int64, float, int]:
            X[col].fillna(X[col].mean(), inplace=True)
        else:
            X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Unknown", inplace=True)

    # --- Fill target missing values (drop rows where target is NaN) ---
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    # --- One-hot encode categorical features ---
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # --- Ensure all columns are numeric ---
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # --- Encode target if classification (string/object) ---
    label_encoder = None
    if y.dtype == object or str(y.dtype) == "category":
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), index=y.index, name=target_col)

    feature_names = X.columns.tolist()

    return X, y, label_encoder, feature_names, removed


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into train and test sets.

    Args:
        X: Feature matrix
        y: Target vector
        test_size: Fraction of data to use as test set
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def analyze_dataset(df: pd.DataFrame, target_col: str) -> dict:
    """
    Analyze the dataset to determine problem type, class balance, size.

    Args:
        df: DataFrame
        target_col: Target column name

    Returns:
        dict with analysis results
    """
    y = df[target_col].dropna()
    n_rows, n_cols = df.shape
    n_unique_target = y.nunique()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Determine problem type
    if y.dtype == object or n_unique_target <= 20:
        problem_type = "classification"
    else:
        problem_type = "regression"

    # Check class imbalance (classification only)
    is_imbalanced = False
    if problem_type == "classification":
        value_counts = y.value_counts(normalize=True)
        min_ratio = value_counts.min()
        if min_ratio < 0.15:
            is_imbalanced = True

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_features": n_cols - 1,
        "n_unique_target": n_unique_target,
        "numeric_features": [c for c in numeric_cols if c != target_col],
        "categorical_features": [c for c in categorical_cols if c != target_col],
        "problem_type": problem_type,
        "is_imbalanced": is_imbalanced,
        "is_large_dataset": n_rows > 5000,
        "target_distribution": y.value_counts().to_dict() if problem_type == "classification" else {},
    }
