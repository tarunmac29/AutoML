"""
feature_selection.py
SelectKBest feature selection using chi2 (classification) or f_regression (regression).
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, f_regression


def select_features(X: pd.DataFrame, y: pd.Series, problem_type: str, k: int = None):
    """
    Select top K best features using statistical tests.

    Args:
        X: Feature matrix (numeric)
        y: Target vector
        problem_type: 'classification' or 'regression'
        k: Number of top features to select (default: min(10, n_features))

    Returns:
        X_selected (DataFrame), selected_feature_names (list), selector (fitted)
    """
    n_features = X.shape[1]

    if k is None:
        k = min(15, n_features)

    k = min(k, n_features)  # safety clamp

    if problem_type == "classification":
        score_func = f_classif
    else:
        score_func = f_regression

    # Replace any inf or nan before selection
    X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(X_clean, y)

    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()
    X_selected = X_clean[selected_features]

    return X_selected, selected_features, selector


def get_feature_scores(X: pd.DataFrame, selector) -> pd.DataFrame:
    """
    Return a DataFrame of feature names and their selection scores.

    Args:
        X: Original feature DataFrame
        selector: Fitted SelectKBest selector

    Returns:
        DataFrame sorted by score descending
    """
    scores = selector.scores_
    feature_names = X.columns.tolist()

    score_df = pd.DataFrame({
        "Feature": feature_names,
        "Score": scores
    }).sort_values("Score", ascending=False).reset_index(drop=True)

    return score_df
