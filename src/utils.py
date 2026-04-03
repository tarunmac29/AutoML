"""
utils.py
Shared utility functions: visualization helpers, metric formatting, model serialization.
"""

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_histograms(df: pd.DataFrame, max_cols: int = 12) -> plt.Figure:
    """
    Plot histograms for all numeric columns (up to max_cols).

    Returns:
        matplotlib Figure
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    n = len(numeric_cols)
    if n == 0:
        return None

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3))
    axes = np.array(axes).flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=25, color="#5DA5E8", edgecolor="white", linewidth=0.5)
        axes[i].set_title(col, fontsize=10, pad=4)
        axes[i].set_xlabel("")
        axes[i].tick_params(labelsize=8)
        sns.despine(ax=axes[i])

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.patch.set_facecolor("none")
    plt.tight_layout(pad=1.5)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    Plot correlation heatmap for numeric columns.

    Returns:
        matplotlib Figure
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr()
    size = max(6, min(14, corr.shape[0]))
    fig, ax = plt.subplots(figsize=(size, size * 0.8))

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=corr.shape[0] <= 15,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        square=True,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Correlation Heatmap", fontsize=12, pad=10)
    fig.patch.set_facecolor("none")
    plt.tight_layout()
    return fig


def plot_target_distribution(y: pd.Series, problem_type: str) -> plt.Figure:
    """
    Plot distribution of the target variable.

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))

    if problem_type == "classification":
        counts = y.value_counts()
        bars = ax.bar(counts.index.astype(str), counts.values, color="#5DA5E8", edgecolor="white")
        ax.bar_label(bars, padding=3, fontsize=9)
        ax.set_xlabel("Class", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title("Target Class Distribution", fontsize=11)
    else:
        ax.hist(y.dropna(), bins=30, color="#5DA5E8", edgecolor="white")
        ax.set_xlabel(y.name, fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title("Target Distribution", fontsize=11)

    sns.despine(ax=ax)
    fig.patch.set_facecolor("none")
    plt.tight_layout()
    return fig


def plot_feature_importance(fi_df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """
    Horizontal bar chart of feature importances.

    Args:
        fi_df: DataFrame with 'Feature' and 'Importance' columns
        top_n: Show top N features

    Returns:
        matplotlib Figure
    """
    if fi_df is None or fi_df.empty:
        return None

    df_plot = fi_df.head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, len(df_plot) * 0.45)))
    bars = ax.barh(df_plot["Feature"][::-1], df_plot["Importance"][::-1], color="#5DA5E8", edgecolor="white")
    ax.set_xlabel("Importance", fontsize=10)
    ax.set_title("Feature Importances", fontsize=11)
    sns.despine(ax=ax)
    fig.patch.set_facecolor("none")
    plt.tight_layout()
    return fig


# ── Model serialization ───────────────────────────────────────────────────────

def model_to_bytes(model) -> bytes:
    """Serialize model to bytes for Streamlit download button."""
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    return buf.read()


# ── Metric formatting ─────────────────────────────────────────────────────────

def format_metrics(metrics: dict, problem_type: str) -> dict:
    """Format metric values as percentage strings where appropriate."""
    formatted = {}
    pct_keys = {"accuracy", "f1_score", "cv_score", "r2_score"}
    for k, v in metrics.items():
        if k in pct_keys and isinstance(v, (int, float)):
            formatted[k] = f"{v * 100:.2f}%"
        elif isinstance(v, float):
            formatted[k] = f"{v:.4f}"
        else:
            formatted[k] = str(v)
    return formatted
