"""
explainability.py
SHAP-based model explainability — feature importance and summary plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap


def get_shap_explainer(model, X_sample: pd.DataFrame, problem_type: str):
    """
    Create appropriate SHAP explainer for the given model.

    Args:
        model: Trained sklearn model
        X_sample: Sample of training data (for background)
        problem_type: 'classification' or 'regression'

    Returns:
        explainer, shap_values
    """
    model_name = type(model).__name__

    try:
        if "RandomForest" in model_name:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            # For classifiers, shap_values is a list — pick class 1 or flatten
            if isinstance(shap_values, list):
                if len(shap_values) == 2:
                    shap_values = shap_values[1]  # binary: use positive class
                else:
                    shap_values = np.abs(np.array(shap_values)).mean(axis=0)

        else:
            # Linear models — use LinearExplainer or KernelExplainer
            background = shap.maskers.Independent(X_sample, max_samples=min(50, len(X_sample)))
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(X_sample)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

        return explainer, shap_values

    except Exception:
        # Fallback to KernelExplainer (slower but universal)
        background = X_sample.sample(min(30, len(X_sample)), random_state=42)
        explainer = shap.KernelExplainer(
            model.predict_proba if problem_type == "classification" and hasattr(model, "predict_proba")
            else model.predict,
            background
        )
        shap_values = explainer.shap_values(X_sample.iloc[:20])
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        return explainer, shap_values


def plot_shap_summary(shap_values, X_sample: pd.DataFrame) -> plt.Figure:
    """
    Create a SHAP summary (beeswarm) plot.

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_sample, show=False, plot_size=None)
    fig = plt.gcf()
    fig.patch.set_facecolor("none")
    plt.tight_layout()
    return fig


def plot_shap_bar(shap_values, X_sample: pd.DataFrame) -> plt.Figure:
    """
    Create a SHAP bar (mean absolute) feature importance plot.

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, plot_size=None)
    fig = plt.gcf()
    fig.patch.set_facecolor("none")
    plt.tight_layout()
    return fig


def get_shap_feature_importance(shap_values, feature_names: list) -> pd.DataFrame:
    """
    Compute mean absolute SHAP values per feature.

    Returns:
        DataFrame with Feature, SHAP_Importance columns
    """
    mean_shap = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Importance": mean_shap
    }).sort_values("SHAP_Importance", ascending=False).reset_index(drop=True)
    return df
