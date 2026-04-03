"""
advanced_viz.py
Advanced charts: ROC Curve, Confusion Matrix, Residual Plots, Model Comparison Dashboard.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score
)


# ── Classification Charts ─────────────────────────────────────────────────────

def plot_confusion_matrix(model, X_test, y_test, label_encoder=None) -> plt.Figure:
    """
    Plot confusion matrix for a classification model.

    Args:
        model: Trained classifier
        X_test: Test features
        y_test: True labels
        label_encoder: Optional LabelEncoder to decode class names

    Returns:
        matplotlib Figure
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    classes = None
    if label_encoder is not None:
        classes = label_encoder.classes_.astype(str)

    fig, ax = plt.subplots(figsize=(max(5, len(np.unique(y_test)) * 1.2), max(4, len(np.unique(y_test)))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(
        ax=ax,
        colorbar=True,
        cmap="Blues",
        values_format="d",
    )
    ax.set_title("Confusion Matrix", fontsize=12, pad=10)
    fig.patch.set_facecolor("none")
    plt.tight_layout()
    return fig


def plot_roc_curve(model, X_test, y_test, label_encoder=None) -> plt.Figure:
    """
    Plot ROC curve (binary or macro-average for multiclass).

    Args:
        model: Trained classifier with predict_proba
        X_test: Test features
        y_test: True labels

    Returns:
        matplotlib Figure or None if predict_proba not available
    """
    if not hasattr(model, "predict_proba"):
        return None

    y_proba = model.predict_proba(X_test)
    classes = np.unique(y_test)
    n_classes = len(classes)

    fig, ax = plt.subplots(figsize=(6, 5))

    if n_classes == 2:
        # Binary ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="#5DA5E8", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    else:
        # Multiclass: one-vs-rest per class
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_test, classes=classes)
        colors = plt.cm.Set2(np.linspace(0, 1, n_classes))
        for i, (cls, color) in enumerate(zip(classes, colors)):
            if y_bin.shape[1] > i:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                lbl = label_encoder.inverse_transform([cls])[0] if label_encoder else cls
                ax.plot(fpr, tpr, color=color, lw=1.8, label=f"Class {lbl} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("ROC Curve", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    sns.despine(ax=ax)
    fig.patch.set_facecolor("none")
    plt.tight_layout()
    return fig


def plot_precision_recall(model, X_test, y_test) -> plt.Figure:
    """
    Plot Precision-Recall curve (binary classification).

    Returns:
        matplotlib Figure or None
    """
    if not hasattr(model, "predict_proba"):
        return None
    classes = np.unique(y_test)
    if len(classes) != 2:
        return None  # PR curve shown for binary only

    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, precision, color="#E24B4A", lw=2, label=f"PR (AP = {ap:.3f})")
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title("Precision-Recall Curve", fontsize=12)
    ax.legend(fontsize=9)
    sns.despine(ax=ax)
    fig.patch.set_facecolor("none")
    plt.tight_layout()
    return fig


# ── Regression Charts ─────────────────────────────────────────────────────────

def plot_residuals(model, X_test, y_test) -> plt.Figure:
    """
    Residual plot + Q-Q plot for regression models.

    Returns:
        matplotlib Figure
    """
    y_pred = model.predict(X_test)
    residuals = np.array(y_test) - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, color="#5DA5E8", s=20, edgecolors="none")
    axes[0].axhline(0, color="#E24B4A", lw=1.5, linestyle="--")
    axes[0].set_xlabel("Predicted values", fontsize=10)
    axes[0].set_ylabel("Residuals", fontsize=10)
    axes[0].set_title("Residuals vs Predicted", fontsize=11)
    sns.despine(ax=axes[0])

    # Histogram of residuals
    axes[1].hist(residuals, bins=25, color="#5DA5E8", edgecolor="white", linewidth=0.4)
    axes[1].axvline(0, color="#E24B4A", lw=1.5, linestyle="--")
    axes[1].set_xlabel("Residual", fontsize=10)
    axes[1].set_ylabel("Count", fontsize=10)
    axes[1].set_title("Residual Distribution", fontsize=11)
    sns.despine(ax=axes[1])

    # Actual vs Predicted
    axes[2].scatter(y_test, y_pred, alpha=0.5, color="#1D9E75", s=20, edgecolors="none")
    lims = [min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))]
    axes[2].plot(lims, lims, "k--", lw=1.2, alpha=0.7)
    axes[2].set_xlabel("Actual values", fontsize=10)
    axes[2].set_ylabel("Predicted values", fontsize=10)
    axes[2].set_title("Actual vs Predicted", fontsize=11)
    sns.despine(ax=axes[2])

    fig.patch.set_facecolor("none")
    plt.tight_layout(pad=1.5)
    return fig


# ── Model Comparison Dashboard ────────────────────────────────────────────────

def plot_model_comparison(all_results: list, problem_type: str) -> plt.Figure:
    """
    Side-by-side bar chart comparing all trained models by their primary metric.

    Args:
        all_results: list of dicts from model_trainer (name, metrics, model)
        problem_type: 'classification' or 'regression'

    Returns:
        matplotlib Figure
    """
    names, primary, secondary = [], [], []

    for r in all_results:
        m = r.get("metrics", {})
        if "error" in m:
            continue
        names.append(r["name"].replace("Classifier", "").replace("Regressor", ""))
        if problem_type == "classification":
            primary.append(m.get("f1_score", 0))
            secondary.append(m.get("accuracy", 0))
        else:
            primary.append(max(m.get("r2_score", 0), 0))
            secondary.append(m.get("cv_score", 0))

    if not names:
        return None

    x = np.arange(len(names))
    width = 0.38

    primary_label = "F1 Score" if problem_type == "classification" else "R² Score"
    secondary_label = "Accuracy" if problem_type == "classification" else "CV Score"

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.4), 4.5))
    bars1 = ax.bar(x - width / 2, primary, width, label=primary_label, color="#5DA5E8", edgecolor="white")
    bars2 = ax.bar(x + width / 2, secondary, width, label=secondary_label, color="#1D9E75", edgecolor="white")

    ax.bar_label(bars1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=15, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Model Comparison Dashboard", fontsize=12, pad=8)
    ax.legend(fontsize=9)
    sns.despine(ax=ax)
    fig.patch.set_facecolor("none")
    plt.tight_layout()
    return fig


def plot_cv_comparison(all_results: list) -> plt.Figure:
    """
    Bar chart of cross-validation scores across all models.

    Returns:
        matplotlib Figure
    """
    names, cv_scores = [], []
    for r in all_results:
        m = r.get("metrics", {})
        if "error" in m or "cv_score" not in m:
            continue
        names.append(r["name"].replace("Classifier", "").replace("Regressor", ""))
        cv_scores.append(m["cv_score"])

    if not names:
        return None

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 4))
    colors = ["#667eea" if s == max(cv_scores) else "#c3cfe2" for s in cv_scores]
    bars = ax.bar(names, cv_scores, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("CV Score", fontsize=10)
    ax.set_title("Cross-Validation Score Comparison", fontsize=12)
    plt.xticks(rotation=15, ha="right", fontsize=9)
    sns.despine(ax=ax)
    fig.patch.set_facecolor("none")
    plt.tight_layout()
    return fig
