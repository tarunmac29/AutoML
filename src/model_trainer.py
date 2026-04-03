"""
model_trainer.py
Intelligent model selection, GridSearchCV tuning, cross-validation, and evaluation.
Now includes: SVM, XGBoost, KNN, Decision Tree alongside original models.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
import joblib
import os

# Try importing XGBoost — optional dependency
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ── Parameter grids ──────────────────────────────────────────────────────────

PARAM_GRIDS = {
    "LogisticRegression": {
        "C": [0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"],
    },
    "RandomForestClassifier": {
        "n_estimators": [50, 100],
        "max_depth": [None, 5, 10],
    },
    "SVC": {
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
    },
    "KNeighborsClassifier": {
        "n_neighbors": [3, 5, 7, 11],
        "weights": ["uniform", "distance"],
    },
    "DecisionTreeClassifier": {
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
    },
    "XGBClassifier": {
        "n_estimators": [50, 100],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
    },
    "LinearRegression": {},
    "Ridge": {"alpha": [0.1, 1.0, 10.0, 100.0]},
    "RandomForestRegressor": {
        "n_estimators": [50, 100],
        "max_depth": [None, 5, 10],
    },
    "SVR": {
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
    },
    "KNeighborsRegressor": {
        "n_neighbors": [3, 5, 7, 11],
        "weights": ["uniform", "distance"],
    },
    "DecisionTreeRegressor": {
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
    },
    "XGBRegressor": {
        "n_estimators": [50, 100],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1, 0.2],
    },
}


def get_candidate_models(analysis: dict) -> list:
    """
    Choose candidate models based on dataset analysis.
    Now includes SVM, KNN, Decision Tree, XGBoost alongside original models.

    Rules:
    - Imbalanced classification → class_weight='balanced' where supported
    - Large dataset → skip KNN and SVM (slow on big data)
    - XGBoost added if available

    Args:
        analysis: dict from preprocessing.analyze_dataset()

    Returns:
        list of (name, model_instance) tuples
    """
    problem_type = analysis["problem_type"]
    is_large = analysis["is_large_dataset"]
    is_imbalanced = analysis["is_imbalanced"]

    if problem_type == "classification":
        cw = "balanced" if is_imbalanced else None

        models = [
            ("LogisticRegression", LogisticRegression(class_weight=cw, max_iter=500, random_state=42)),
            ("RandomForestClassifier", RandomForestClassifier(class_weight=cw, random_state=42)),
            ("DecisionTreeClassifier", DecisionTreeClassifier(class_weight=cw, random_state=42)),
        ]
        if not is_large:
            models += [
                ("SVC", SVC(class_weight=cw, probability=True, random_state=42)),
                ("KNeighborsClassifier", KNeighborsClassifier()),
            ]
        if XGBOOST_AVAILABLE:
            # XGBoost uses scale_pos_weight for imbalance, not class_weight
            models.append(("XGBClassifier", XGBClassifier(
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, verbosity=0
            )))

    else:
        models = [
            ("LinearRegression", LinearRegression()),
            ("Ridge", Ridge()),
            ("RandomForestRegressor", RandomForestRegressor(random_state=42)),
            ("DecisionTreeRegressor", DecisionTreeRegressor(random_state=42)),
        ]
        if not is_large:
            models += [
                ("SVR", SVR()),
                ("KNeighborsRegressor", KNeighborsRegressor()),
            ]
        if XGBOOST_AVAILABLE:
            models.append(("XGBRegressor", XGBRegressor(random_state=42, verbosity=0)))

    return models


def train_and_select_best(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    analysis: dict,
    cv: int = 3,
) -> dict:
    """
    Train all candidate models with GridSearchCV, evaluate, and pick the best.

    Args:
        X_train, X_test, y_train, y_test: Split data
        analysis: Dataset analysis dict
        cv: Cross-validation folds

    Returns:
        dict with best_model, best_name, metrics, all_results
    """
    problem_type = analysis["problem_type"]
    candidates = get_candidate_models(analysis)
    scoring = "f1_weighted" if problem_type == "classification" else "r2"

    all_results = []
    best_score = -np.inf
    best_model = None
    best_name = None
    best_metrics = {}

    for name, model in candidates:
        param_grid = PARAM_GRIDS.get(name, {})

        try:
            if param_grid:
                grid_search = GridSearchCV(
                    model,
                    param_grid,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1,
                    error_score="raise",
                )
                grid_search.fit(X_train, y_train)
                fitted_model = grid_search.best_estimator_
                cv_score = grid_search.best_score_
            else:
                # Linear Regression — no grid to search
                model.fit(X_train, y_train)
                fitted_model = model
                cv_scores = cross_val_score(fitted_model, X_train, y_train, cv=cv, scoring=scoring)
                cv_score = cv_scores.mean()

            y_pred = fitted_model.predict(X_test)

            if problem_type == "classification":
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                metrics = {"accuracy": round(acc, 4), "f1_score": round(f1, 4), "cv_score": round(cv_score, 4)}
                compare_score = f1
            else:
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                metrics = {"r2_score": round(r2, 4), "rmse": round(rmse, 4), "cv_score": round(cv_score, 4)}
                compare_score = r2

            all_results.append({"name": name, "metrics": metrics, "model": fitted_model})

            if compare_score > best_score:
                best_score = compare_score
                best_model = fitted_model
                best_name = name
                best_metrics = metrics

        except Exception as e:
            all_results.append({"name": name, "metrics": {"error": str(e)}, "model": None})

    return {
        "best_model": best_model,
        "best_name": best_name,
        "best_metrics": best_metrics,
        "all_results": all_results,
        "problem_type": problem_type,
    }


def save_model(model, path: str = "models/model.pkl"):
    """Save the trained model to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str = "models/model.pkl"):
    """Load a saved model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at '{path}'.")
    return joblib.load(path)


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importances from tree-based models or coefficients from linear.

    Args:
        model: Trained model
        feature_names: List of feature names

    Returns:
        DataFrame sorted by importance descending
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(coef).flatten()[:len(feature_names)]
    else:
        return pd.DataFrame()

    fi_df = pd.DataFrame({
        "Feature": feature_names[:len(importances)],
        "Importance": importances
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return fi_df
