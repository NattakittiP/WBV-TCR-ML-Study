"""
research_New.py — Main leakage-controlled ML pipeline for WBV–TCR study.

This script:
- Loads and preprocesses the clinical dataset
- Builds leakage-controlled ML models (LogReg, RF, XGB)
- Performs nested cross-validation (RF, XGB)
- Calibrates models with isotonic regression
- Evaluates test performance with Youden threshold
- Computes bootstrap 95% CI for ROC-AUC
- Plots ROC, PR, and calibration curves
- Computes permutation feature importance (RF)
- Runs permutation-based SHAP for XGB
- Runs ablation experiments (TG-only vs TG+WBV/HCT/TP)
- Generates simple 0–100 risk points from logistic regression

Outputs are saved in the `outputs_pro/` directory.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

import xgboost as xgb
import shap

from utils import (
    set_global_seed,
    ensure_dir,
    save_json,
    evaluate_model,
    bootstrap_auc_ci,
    plot_roc_curves,
    plot_pr_curves,
    plot_calibration,
    nested_cv_auc,
    run_ablation_experiments,
)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


CSV_PATH = r"C:\Users\Sabi\Desktop\Ohm_files\WBV_TCR_ICBCB2026_Synthetic_1500_precise_v2 (2).csv"

FEATURES = ["WBV", "Hematocrit", "TotalProtein", "TG0h", "TG4h",
            "Age", "Sex", "HDL", "LDL", "BMI"]

SEED = 42
TEST_SIZE = 0.20
N_SPLITS_OUTER = 5
N_SPLITS_INNER = 5
N_ITERS_TUNE = 40  # random search iterations
OUTDIR = ensure_dir("outputs_pro")


# ---------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------

def load_and_prepare_data(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load dataset, preprocess features, and construct binary target from TCR.

    Target rule:
        - If column 'y' exists: use it directly (0/1)
        - Else: y = 1 if TCR <= Q1(TCR), 0 otherwise
    """
    df = pd.read_csv(csv_path)

    if not {"TCR", "Sex"}.issubset(df.columns):
        raise ValueError("Dataset must contain at least 'TCR' and 'Sex' columns.")

    # Features
    X = df[FEATURES].copy()

    # Encode Sex if string
    if X["Sex"].dtype == object:
        X["Sex"] = (
            X["Sex"].astype(str)
            .str.strip()
            .str.lower()
            .map({"m": 1, "male": 1, "1": 1, "f": 0, "female": 0, "0": 0})
        )

    # Force numeric + median imputation
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median())

    # Target
    if "y" in df.columns:
        y = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int)
    else:
        q1 = df["TCR"].quantile(0.25)
        y = (df["TCR"] <= q1).astype(int)

    return X, y


# ---------------------------------------------------------------------
# Model definitions & search spaces
# ---------------------------------------------------------------------

def build_models_and_spaces():
    """Create base models and hyperparameter search spaces."""
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=5000,
                                  class_weight="balanced",
                                  random_state=SEED)),
    ])

    rf = RandomForestClassifier(
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    )
    rf_space = {
        "n_estimators": [200, 400, 600, 800, 1000],
        "max_depth": [None, 3, 5, 7, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }

    xgb_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        scale_pos_weight=1.0,
    )
    xgb_space = {
        "n_estimators": [200, 400, 600, 800],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.3],
    }

    return lr, rf, xgb_clf, rf_space, xgb_space


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main():
    set_global_seed(SEED)

    # 1) Load & split data ---------------------------------------------------
    X, y = load_and_prepare_data(CSV_PATH)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    # 2) Models & search spaces ---------------------------------------------
    lr_base, rf_base, xgb_base, rf_space, xgb_space = build_models_and_spaces()

    # 3) Nested CV (RF, XGB) ------------------------------------------------
    # Use utils.nested_cv_auc with ndarray to avoid indexing issues
    rf_cv = nested_cv_auc(
        rf_base,
        rf_space,
        X_tr.values,
        y_tr.values,
        outer_splits=N_SPLITS_OUTER,
        inner_splits=N_SPLITS_INNER,
        iters=N_ITERS_TUNE,
        seed=SEED,
        scoring="roc_auc",
    )
    xgb_cv = nested_cv_auc(
        xgb_base,
        xgb_space,
        X_tr.values,
        y_tr.values,
        outer_splits=N_SPLITS_OUTER,
        inner_splits=N_SPLITS_INNER,
        iters=N_ITERS_TUNE,
        seed=SEED,
        scoring="roc_auc",
    )

    nested_summary = {
        "RF_auc_mean": rf_cv["auc_mean"],
        "RF_auc_std": rf_cv["auc_std"],
        "RF_auc_all": rf_cv["auc_all"],
        "RF_best_params": rf_cv["best_params"],
        "XGB_auc_mean": xgb_cv["auc_mean"],
        "XGB_auc_std": xgb_cv["auc_std"],
        "XGB_auc_all": xgb_cv["auc_all"],
        "XGB_best_params": xgb_cv["best_params"],
    }
    save_json(nested_summary, OUTDIR / "nested_cv_summary.json")

    print(f"[NestedCV] RF  AUC {rf_cv['auc_mean']:.3f} ± {rf_cv['auc_std']:.3f}")
    print(f"[NestedCV] XGB AUC {xgb_cv['auc_mean']:.3f} ± {xgb_cv['auc_std']:.3f}")

    # 4) Fit best RF & XGB on full train via RandomizedSearchCV --------------
    inner_cv = StratifiedKFold(
        n_splits=N_SPLITS_INNER,
        shuffle=True,
        random_state=SEED,
    )

    rf_search = RandomizedSearchCV(
        rf_base,
        rf_space,
        n_iter=N_ITERS_TUNE,
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=-1,
        random_state=SEED,
        refit=True,
    ).fit(X_tr, y_tr)
    rf_best = rf_search.best_estimator_

    xgb_search = RandomizedSearchCV(
        xgb_base,
        xgb_space,
        n_iter=N_ITERS_TUNE,
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=-1,
        random_state=SEED,
        refit=True,
    ).fit(X_tr, y_tr)
    xgb_best = xgb_search.best_estimator_

    # 5) Calibration (Isotonic) + Logistic baseline -------------------------
    rf_cal = CalibratedClassifierCV(
        rf_best,
        method="isotonic",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    ).fit(X_tr, y_tr)

    xgb_cal = CalibratedClassifierCV(
        xgb_best,
        method="isotonic",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    ).fit(X_tr, y_tr)

    lr_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=5000,
                                  class_weight="balanced",
                                  random_state=SEED)),
    ]).fit(X_tr, y_tr)

    models = {
        "LogReg": lr_clf,
        "RF_cal": rf_cal,
        "XGB_cal": xgb_cal,
    }

    # 6) Test metrics + Youden threshold ------------------------------------
    test_report: dict[str, dict] = {}
    for name, model in models.items():
        res = evaluate_model(model, X_te, y_te, threshold="youden")
        # ไม่จำเป็นต้องเก็บ proba ลง CSV
        res_clean = {k: v for k, v in res.items() if k != "proba"}
        test_report[name] = res_clean

    pd.DataFrame(test_report).to_csv(OUTDIR / "test_report.csv")

    # 7) Bootstrap 95% CI for ROC-AUC ---------------------------------------
    auc_ci = {}
    for name, model in models.items():
        low, high = bootstrap_auc_ci(model, X_te, y_te, B=1000, seed=SEED)
        auc_ci[name] = [low, high]

    save_json(auc_ci, OUTDIR / "auc_ci.json")

    # 8) ROC, PR, Calibration plots -----------------------------------------
    plot_roc_curves(
        models,
        X_te,
        y_te,
        out_path=OUTDIR / "calibrated_roc.png",
        title="ROC (Test, calibrated models)",
    )
    plot_pr_curves(
        models,
        X_te,
        y_te,
        out_path=OUTDIR / "calibrated_pr.png",
        title="Precision–Recall (Test, calibrated models)",
    )
    plot_calibration(
        models,
        X_te,
        y_te,
        n_bins=10,
        out_path=OUTDIR / "calibration_curves.png",
        title="Calibration curves (Test)",
    )

    # 9) Permutation importance (RF) ----------------------------------------
    rf_fitted = rf_best.fit(X_tr, y_tr)
    perm = permutation_importance(
        rf_fitted,
        X_te,
        y_te,
        scoring="roc_auc",
        n_repeats=20,
        random_state=SEED,
    )
    imp = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
    imp.to_csv(OUTDIR / "perm_importance_rf.csv")

    # 10) Permutation-SHAP for XGB ------------------------------------------
    try:
        explainer = shap.Explainer(
            xgb_best.predict_proba,
            X_te,
            algorithm="permutation",
            feature_names=X_te.columns,
        )
        sv = explainer(X_te)

        # Global importance (beeswarm)
        shap.plots.beeswarm(sv, show=False)
        plt.tight_layout()
        plt.savefig(OUTDIR / "shap_beeswarm_xgb_perm.png", dpi=300)
        plt.close()

        # Dependence plots for key features
        for feat in ["TG4h", "TG0h", "WBV"]:
            if feat in X_te.columns:
                shap.plots.scatter(sv[:, feat], color=sv, show=False)
                plt.tight_layout()
                plt.savefig(OUTDIR / f"shap_perm_depend_{feat}.png", dpi=300)
                plt.close()
        print("[INFO] Saved permutation-SHAP plots.")
    except Exception as e:
        print("[WARN] Permutation SHAP failed:", repr(e))

    # 11) Ablation: TG-only vs TG+WBV/HCT/TP --------------------------------
    feature_sets = {
        "TG-only": ["TG0h", "TG4h"],
        "TG+WBV/HCT/TP": ["TG0h", "TG4h", "WBV", "Hematocrit", "TotalProtein"],
    }
    ablation_df = run_ablation_experiments(
        X,
        y,
        feature_sets=feature_sets,
        test_size=TEST_SIZE,
        seed=SEED,
    )
    
    with open(OUTDIR / "ablation.txt", "w", encoding="utf-8") as f:
        for name, row in ablation_df.iterrows():
            f.write(f"{name}: AUC={row['AUC']:.3f}, PR-AUC={row['PR_AUC']:.3f}\n")

    # 12) Risk score from logistic regression -------------------------------
    scaler = StandardScaler().fit(X_tr)
    Xtr_s = scaler.transform(X_tr)
    Xte_s = scaler.transform(X_te)

    lr_final = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=SEED,
    ).fit(Xtr_s, y_tr)

    coefs = pd.Series(lr_final.coef_[0], index=X.columns)
    coefs.to_csv(OUTDIR / "logistic_coefficients.csv")

    
    weights = (coefs / np.sum(np.abs(coefs))).fillna(0)
    risk_points = (weights.abs() / weights.abs().max() * 100).round().astype(int)
    risk_points.sort_values(ascending=False).to_csv(OUTDIR / "risk_points_0_100.csv")

    print("[DONE] All results saved to", OUTDIR.resolve())


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()
"""

