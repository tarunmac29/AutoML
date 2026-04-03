"""
feature_engineering.py
Auto Feature Engineering: polynomial features, log transforms, interaction terms.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def apply_log_transform(X: pd.DataFrame, threshold: float = 0.0) -> tuple:
    """
    Apply log1p transform to right-skewed numeric columns.
    Skewness > threshold → apply log1p.

    Args:
        X: Feature DataFrame
        threshold: Skewness threshold (default 0.5)

    Returns:
        (transformed DataFrame, list of transformed column names)
    """
    X = X.copy()
    transformed_cols = []

    for col in X.select_dtypes(include=[np.number]).columns:
        if X[col].min() >= 0:  # log only valid for non-negative
            skew = X[col].skew()
            if abs(skew) > threshold:
                X[col] = np.log1p(X[col])
                transformed_cols.append(col)

    return X, transformed_cols


def apply_polynomial_features(X: pd.DataFrame, degree: int = 2, top_n: int = 5) -> tuple:
    """
    Generate polynomial + interaction features for top N numeric columns.
    Limits to top_n columns to avoid feature explosion.

    Args:
        X: Feature DataFrame
        degree: Polynomial degree (2 recommended)
        top_n: Max columns to expand (keeps feature count manageable)

    Returns:
        (expanded DataFrame, list of new feature names)
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()[:top_n]
    if len(numeric_cols) < 2:
        return X, []

    X_num = X[numeric_cols]
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    X_poly = poly.fit_transform(X_num)

    poly_feature_names = poly.get_feature_names_out(numeric_cols)
    new_cols = [c for c in poly_feature_names if c not in numeric_cols]

    X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=X.index)
    new_features_df = X_poly_df[new_cols]

    # Drop original numeric cols from X, replace with poly version
    X_out = X.drop(columns=numeric_cols).join(X_poly_df[poly_feature_names])

    return X_out, new_cols


def apply_interaction_terms(X: pd.DataFrame, top_n: int = 6) -> tuple:
    """
    Generate pairwise interaction (multiplication) features for top N columns.

    Args:
        X: Feature DataFrame
        top_n: Max columns to consider

    Returns:
        (DataFrame with interactions added, list of new column names)
    """
    X = X.copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()[:top_n]
    new_cols = []

    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            c1, c2 = numeric_cols[i], numeric_cols[j]
            new_name = f"{c1}_x_{c2}"
            X[new_name] = X[c1] * X[c2]
            new_cols.append(new_name)

    return X, new_cols


def engineer_features(X: pd.DataFrame, config: dict) -> tuple:
    """
    Master feature engineering pipeline.

    Config keys:
    - log_transform (bool): Apply log1p to skewed columns
    - polynomial (bool): Add polynomial features
    - poly_degree (int): Degree for polynomial (default 2)
    - interactions (bool): Add interaction terms
    - top_n (int): Max columns for poly/interactions (default 5)

    Args:
        X: Feature DataFrame
        config: Dict of enabled transformations

    Returns:
        (transformed DataFrame, summary dict)
    """
    summary = {}
    X = X.copy()

    if config.get("log_transform", False):
        X, log_cols = apply_log_transform(X)
        summary["log_transformed"] = log_cols

    if config.get("interactions", False) and not config.get("polynomial", False):
        top_n = config.get("top_n", 5)
        X, inter_cols = apply_interaction_terms(X, top_n=top_n)
        summary["interaction_features"] = inter_cols

    if config.get("polynomial", False):
        top_n = config.get("top_n", 5)
        degree = config.get("poly_degree", 2)
        X, poly_cols = apply_polynomial_features(X, degree=degree, top_n=top_n)
        summary["polynomial_features"] = poly_cols

    # Final cleanup
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    summary["total_features_after"] = X.shape[1]

    return X, summary
