"""
utils.py — Utility functions for leakage-controlled clinical ML pipelines
Author: Nattakitti Piyavechvirat (Ohm) - WBV–TCR study

This module centralizes reusable helpers for:
- Reproducible seeding
- Directory handling
- Model evaluation (ROC-AUC, PR-AUC, Brier, F1, ACC, CM)
- Bootstrap confidence intervals
- ROC / PR / calibration plotting
- Nested cross-validation
- Quick ablation experiments
- JSON I/O helpers

Designed to be used together with research.py / research_New.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def set_global_seed(seed: int = 42) -> None:
    """
    Set global random seed for numpy and Python's RNG (if needed).
    """
    import random

    np.random.seed(seed)
    random.seed(seed)


def ensure_dir(path: Path | str) -> Path:
    """
    Ensure a directory exists. Returns the Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: dict, path: Path | str, indent: int = 2) -> None:
    """
    Save a Python dict as JSON.
    """
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)


def load_json(path: Path | str) -> dict:
    """
    Load JSON file into a Python dict.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Model evaluation utilities
# ---------------------------------------------------------------------------

def evaluate_model(
    model: BaseEstimator,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    threshold: float | str = "youden",
) -> Dict[str, object]:
    """
    Evaluate a binary classifier with probabilistic output.

    Parameters
    ----------
    model : fitted estimator with predict_proba
    X : features (DataFrame or ndarray)
    y : true labels (0/1)
    threshold : float or "youden"
        If "youden", use Youden's J statistic to pick optimal threshold.
        If float, use that fixed threshold.

    Returns
    -------
    metrics : dict
        Contains:
        - threshold
        - roc_auc
        - pr_auc
        - f1
        - acc
        - brier
        - confusion_matrix
        - proba (raw predicted probabilities)
    """
    y = np.asarray(y).astype(int)
    proba = model.predict_proba(X)[:, 1]

    thr = threshold
    if threshold == "youden":
        fpr, tpr, th = roc_curve(y, proba)
        j = tpr - fpr
        thr = float(th[np.argmax(j)])

    preds = (proba >= thr).astype(int)

    cm = confusion_matrix(y, preds)
    metrics = {
        "threshold": float(thr),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "f1": float(f1_score(y, preds)),
        "acc": float(accuracy_score(y, preds)),
        "brier": float(brier_score_loss(y, proba)),
        "cm": cm.tolist(),
        "proba": proba,  # keep as np.ndarray (not JSON safe, but useful in memory)
    }
    return metrics


def evaluate_models_dict(
    models: Dict[str, BaseEstimator],
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    threshold: float | str = "youden",
) -> pd.DataFrame:
    """
    Evaluate multiple models and return a summary DataFrame.

    Parameters
    ----------
    models : dict
        Mapping from model name -> fitted estimator.
    X, y : data + labels
    threshold : float or "youden"

    Returns
    -------
    df : DataFrame
        Rows = metrics, Columns = model names.
    """
    records = {}
    for name, m in models.items():
        m_res = evaluate_model(m, X, y, threshold=threshold)
        # drop proba & cm from summary; keep scalar metrics
        rec = {
            "threshold": m_res["threshold"],
            "roc_auc": m_res["roc_auc"],
            "pr_auc": m_res["pr_auc"],
            "f1": m_res["f1"],
            "acc": m_res["acc"],
            "brier": m_res["brier"],
        }
        records[name] = rec
    return pd.DataFrame(records)


def bootstrap_auc_ci(
    model: BaseEstimator,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    B: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Bootstrap 95% confidence interval for ROC-AUC on a fixed test set.

    Parameters
    ----------
    model : fitted estimator
    X, y : test data
    B : number of bootstrap iterations
    seed : random seed

    Returns
    -------
    (low, high) : tuple of floats
        2.5th and 97.5th percentiles of bootstrapped AUCs.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    proba = model.predict_proba(X)[:, 1]
    n = len(y)

    aucs: List[float] = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        aucs.append(roc_auc_score(y[idx], proba[idx]))

    low, high = np.percentile(aucs, [2.5, 97.5])
    return float(low), float(high)


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_roc_curves(
    models: Dict[str, BaseEstimator],
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    out_path: Optional[Path | str] = None,
    title: str = "ROC curves",
    dpi: int = 300,
) -> None:
    """
    Plot ROC curves for multiple models on the same figure.
    """
    plt.figure()
    y = np.asarray(y).astype(int)

    for name, m in models.items():
        proba = m.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, proba)
        auc = roc_auc_score(y, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_pr_curves(
    models: Dict[str, BaseEstimator],
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    out_path: Optional[Path | str] = None,
    title: str = "Precision–Recall curves",
    dpi: int = 300,
) -> None:
    """
    Plot Precision–Recall curves for multiple models.
    """
    plt.figure()
    y = np.asarray(y).astype(int)

    for name, m in models.items():
        proba = m.predict_proba(X)[:, 1]
        prec, rec, _ = precision_recall_curve(y, proba)
        pr_auc = average_precision_score(y, proba)
        plt.plot(rec, prec, label=f"{name} (PR-AUC={pr_auc:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_calibration(
    models: Dict[str, BaseEstimator],
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    n_bins: int = 10,
    out_path: Optional[Path | str] = None,
    title: str = "Calibration curves",
    dpi: int = 300,
) -> None:
    """
    Plot calibration curves (reliability diagrams) for multiple models.
    """
    plt.figure()
    y = np.asarray(y).astype(int)

    for name, m in models.items():
        proba = m.predict_proba(X)[:, 1]
        frac_pos, mean_pred = calibration_curve(y, proba, n_bins=n_bins, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", linestyle="-", label=name)

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.legend(loc="upper left")
    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Iterable[str] = ("Negative", "Positive"),
    normalize: bool = False,
    out_path: Optional[Path | str] = None,
    title: str = "Confusion matrix",
    dpi: int = 300,
) -> None:
    """
    Simple confusion matrix heatmap.

    Parameters
    ----------
    cm : confusion matrix (2x2)
    normalize : whether to normalize rows to proportions
    """
    cm = np.asarray(cm)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure()
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(title)
    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=dpi)
    plt.close()


# ---------------------------------------------------------------------------
# Nested cross-validation utility
# ---------------------------------------------------------------------------

def nested_cv_auc(
    model: BaseEstimator,
    param_space: dict,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    outer_splits: int = 5,
    inner_splits: int = 5,
    iters: int = 40,
    seed: int = 42,
    scoring: str = "roc_auc",
) -> Dict[str, object]:
    """
    Perform nested cross-validation with RandomizedSearchCV.

    Parameters
    ----------
    model : estimator (e.g., RandomForestClassifier, XGBClassifier)
    param_space : dict
        Hyperparameter search space for RandomizedSearchCV.
    X, y : data
    outer_splits : number of outer folds
    inner_splits : number of inner folds
    iters : number of RandomizedSearch iterations per outer fold
    seed : random seed
    scoring : metric for optimization (default "roc_auc")

    Returns
    -------
    result : dict with keys
        - auc_mean
        - auc_std
        - auc_all (list of per-fold AUCs)
        - best_params (list of best_params per outer fold)
    """
    y = np.asarray(y).astype(int)
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)

    aucs: List[float] = []
    best_params_all: List[dict] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(outer.split(X, y), start=1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
        search = RandomizedSearchCV(
            model,
            param_distributions=param_space,
            n_iter=iters,
            scoring=scoring,
            cv=inner,
            n_jobs=-1,
            random_state=seed,
            refit=True,
        )
        search.fit(X_tr, y_tr)
        best_est = search.best_estimator_
        proba = best_est.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, proba)

        aucs.append(float(auc))
        best_params_all.append(search.best_params_)

        print(f"[NestedCV] Fold {fold_idx}/{outer_splits} AUC={auc:.3f}")

    auc_mean = float(np.mean(aucs))
    auc_std = float(np.std(aucs))

    return {
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "auc_all": aucs,
        "best_params": best_params_all,
    }


# ---------------------------------------------------------------------------
# Ablation helper
# ---------------------------------------------------------------------------

def quick_xgb_auc(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Quick XGBoost AUC/PR-AUC on a single train/validation split.
    Used for ablation comparisons.

    NOTE: This requires xgboost to be installed.
    """
    from sklearn.model_selection import train_test_split
    import xgboost as xgb

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=seed,
    )
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, proba)
    pr = average_precision_score(y_va, proba)
    return float(auc), float(pr)


def run_ablation_experiments(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    feature_sets: Dict[str, List[str]],
    test_size: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run simple ablation experiments comparing different feature subsets
    using quick_xgb_auc.

    Parameters
    ----------
    X : full feature DataFrame
    y : labels
    feature_sets : dict
        Mapping from label -> list of column names.
        e.g. {
            "TG-only": ["TG0h", "TG4h"],
            "TG+WBV/HCT/TP": ["TG0h", "TG4h", "WBV", "Hematocrit", "TotalProtein"]
        }
    test_size : fraction for validation
    seed : random seed

    Returns
    -------
    df : DataFrame with columns ["AUC", "PR_AUC"] indexed by ablation name.
    """
    records = {}
    for name, cols in feature_sets.items():
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"[Ablation] Missing columns for {name}: {missing}")

        auc, pr = quick_xgb_auc(X[cols], y, test_size=test_size, seed=seed)
        records[name] = {"AUC": auc, "PR_AUC": pr}
        print(f"[Ablation] {name}: AUC={auc:.3f}, PR-AUC={pr:.3f}")

    return pd.DataFrame(records).T
