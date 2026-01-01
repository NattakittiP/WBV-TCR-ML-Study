# research_pro.py  — deep ML pipeline for your dataset
import os, sys, warnings, json, math
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, accuracy_score,
                             brier_score_loss, roc_curve, precision_recall_curve, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

import xgboost as xgb
import shap

# ------------ Config ------------
CSV_PATH = r"C:\Users\Sabi\Desktop\Ohm_files\WBV_TCR_ICBCB2026_Synthetic_1500_precise_v2 (2).csv"
FEATURES = ["WBV","Hematocrit","TotalProtein","TG0h","TG4h","Age","Sex","HDL","LDL","BMI"]
SEED = 42
TEST_SIZE = 0.20
N_SPLITS_OUTER = 5
N_SPLITS_INNER = 5
N_ITERS_TUNE = 40  # random search iterations
OUTDIR = Path("outputs_pro"); OUTDIR.mkdir(exist_ok=True)

# ------------ Load & target ------------
df = pd.read_csv(CSV_PATH)
assert set(["TCR","Sex"]).issubset(df.columns), "ต้องมีคอลัมน์ TCR และ Sex"
X = df[FEATURES].copy()
# encode Sex if needed
if X["Sex"].dtype == object:
    X["Sex"] = (X["Sex"].astype(str).str.strip().str.lower()
                .map({"m":1,"male":1,"1":1,"f":0,"female":0,"0":0}))
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median())

# target: ใช้ y ถ้ามี ไม่งั้นสร้างจาก TCR<=Q1
if "y" in df.columns:
    y = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype(int)
else:
    q1 = df["TCR"].quantile(0.25)
    y = (df["TCR"] <= q1).astype(int)

# ------------ Hold-out split ------------
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                          random_state=SEED, stratify=y)

# ------------ Models & search spaces ------------
lr = Pipeline([("scaler", StandardScaler()),
               ("lr", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED))])

rf = RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=SEED)
rf_space = {
    "n_estimators": [200,400,600,800,1000],
    "max_depth": [None, 3,5,7,10,15],
    "min_samples_split": [2,5,10],
    "min_samples_leaf": [1,2,4],
    "max_features": ["sqrt", "log2", None]
}

xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic", eval_metric="logloss",
    tree_method="hist", random_state=SEED, n_jobs=-1, scale_pos_weight=1.0
)
xgb_space = {
    "n_estimators": [200,400,600,800],
    "max_depth": [2,3,4,5,6],
    "learning_rate": [0.01,0.03,0.05,0.1],
    "subsample": [0.6,0.8,1.0],
    "colsample_bytree": [0.6,0.8,1.0],
    "min_child_weight": [1,3,5],
    "gamma": [0, 0.1, 0.3]
}

# ------------ Nested CV for RF & XGB (optimize ROC-AUC) ------------
def nested_cv(model, param_space, X, y, outer_splits=5, inner_splits=5, iters=40):
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=SEED)
    scores = []; best_params = []
    for tr_idx, va_idx in outer.split(X, y):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=SEED)
        search = RandomizedSearchCV(
            model, param_space, n_iter=iters, scoring="roc_auc",
            cv=inner, n_jobs=-1, random_state=SEED, refit=True
        )
        search.fit(Xtr, ytr)
        proba = search.best_estimator_.predict_proba(Xva)[:,1]
        scores.append(roc_auc_score(yva, proba))
        best_params.append(search.best_params_)
    return np.mean(scores), np.std(scores), best_params

rf_auc_m, rf_auc_s, rf_params = nested_cv(rf, rf_space, X_tr, y_tr,
                                          outer_splits=N_SPLITS_OUTER, inner_splits=N_SPLITS_INNER, iters=N_ITERS_TUNE)
xgb_auc_m, xgb_auc_s, xgb_params = nested_cv(xgb_clf, xgb_space, X_tr, y_tr,
                                            outer_splits=N_SPLITS_OUTER, inner_splits=N_SPLITS_INNER, iters=N_ITERS_TUNE)

with open(OUTDIR/"nested_cv_summary.json","w") as f:
    json.dump({"RF_auc_mean":rf_auc_m, "RF_auc_std":rf_auc_s, "RF_best_params":rf_params,
               "XGB_auc_mean":xgb_auc_m,"XGB_auc_std":xgb_auc_s,"XGB_best_params":xgb_params}, f, indent=2)

print(f"[NestedCV] RF AUC {rf_auc_m:.3f}±{rf_auc_s:.3f} | XGB AUC {xgb_auc_m:.3f}±{xgb_auc_s:.3f}")

# ------------ Fit best models on full train (inner refit) ------------
rf_best = RandomizedSearchCV(rf, rf_space, n_iter=N_ITERS_TUNE, scoring="roc_auc",
                             cv=StratifiedKFold(n_splits=N_SPLITS_INNER, shuffle=True, random_state=SEED),
                             n_jobs=-1, random_state=SEED, refit=True).fit(X_tr, y_tr).best_estimator_
xgb_best = RandomizedSearchCV(xgb_clf, xgb_space, n_iter=N_ITERS_TUNE, scoring="roc_auc",
                              cv=StratifiedKFold(n_splits=N_SPLITS_INNER, shuffle=True, random_state=SEED),
                              n_jobs=-1, random_state=SEED, refit=True).fit(X_tr, y_tr).best_estimator_

# ------------ Calibration (Isotonic) ------------
rf_cal = CalibratedClassifierCV(rf_best, method="isotonic", cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)).fit(X_tr, y_tr)
xgb_cal = CalibratedClassifierCV(xgb_best, method="isotonic", cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)).fit(X_tr, y_tr)
lr_clf = Pipeline([("scaler", StandardScaler()),
                   ("lr", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED))]).fit(X_tr, y_tr)

models = {"LogReg": lr_clf, "RF_cal": rf_cal, "XGB_cal": xgb_cal}

# ------------ Test metrics + threshold tuning ------------
def eval_model(m, X, y, thr=0.5):
    proba = m.predict_proba(X)[:,1]
    if thr=="youden":
        fpr, tpr, th = roc_curve(y, proba); j = tpr - fpr; thr = th[np.argmax(j)]
    preds = (proba >= thr).astype(int)
    return dict(
        thr=thr,
        roc=roc_auc_score(y, proba),
        pr=average_precision_score(y, proba),
        f1=f1_score(y, preds),
        acc=accuracy_score(y, preds),
        brier=brier_score_loss(y, proba),
        cm=confusion_matrix(y, preds),
        proba=proba
    )

test_report = {name: eval_model(m, X_te, y_te, thr="youden") for name,m in models.items()}
pd.DataFrame(test_report).to_csv(OUTDIR/"test_report.csv")

# ------------ Bootstrap 95% CI (ROC-AUC) ------------
def bootstrap_ci(model, X, y, B=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    aucs=[]
    proba = model.predict_proba(X)[:,1]
    for _ in range(B):
        idx = rng.integers(0, len(y), len(y))
        aucs.append(roc_auc_score(y.iloc[idx], proba[idx]))
    return np.percentile(aucs,[2.5,97.5])

ci = {name: bootstrap_ci(m, X_te, y_te, B=1000) for name,m in models.items()}
with open(OUTDIR/"auc_ci.json","w") as f: json.dump({k:list(map(float,v)) for k,v in ci.items()}, f, indent=2)

# ------------ Plots: ROC, PR, Calibration ------------
def plot_roc_pr(models, X, y, out_prefix):
    plt.figure()
    for k,m in models.items():
        proba = m.predict_proba(X)[:,1]
        fpr,tpr,_ = roc_curve(y, proba)
        plt.plot(fpr,tpr,label=f"{k} (AUC={roc_auc_score(y, proba):.2f})")
    plt.plot([0,1],[0,1],"--"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (Test)"); plt.legend()
    plt.tight_layout(); plt.savefig(OUTDIR/f"{out_prefix}_roc.png",dpi=300); plt.close()

    plt.figure()
    for k,m in models.items():
        proba = m.predict_proba(X)[:,1]
        prec,rec,_ = precision_recall_curve(y, proba)
        plt.plot(rec,prec,label=f"{k} (PR-AUC={average_precision_score(y, proba):.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR (Test)"); plt.legend()
    plt.tight_layout(); plt.savefig(OUTDIR/f"{out_prefix}_pr.png",dpi=300); plt.close()

plot_roc_pr(models, X_te, y_te, "calibrated")

# ------------ Explainability: Permutation + SHAP ------------
rf_fitted = rf_best.fit(X_tr, y_tr)
perm = permutation_importance(rf_fitted, X_te, y_te, scoring="roc_auc", n_repeats=20, random_state=SEED)
imp = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
imp.to_csv(OUTDIR/"perm_importance_rf.csv")

# SHAP (XGB)
# ------------ SHAP (Permutation-based, version-robust) ------------

try:
    
    explainer = shap.Explainer(
        xgb_best.predict_proba,       
        X_te,                         
        algorithm="permutation",      
        feature_names=X_te.columns
    )
    sv = explainer(X_te)              

    # Beeswarm (global importance)
    shap.plots.beeswarm(sv, show=False)
    plt.tight_layout(); plt.savefig(OUTDIR/"shap_beeswarm_xgb_perm.png", dpi=300); plt.close()

    # Dependence plots เฉพาะฟีเจอร์เด่น
    for feat in ["TG4h", "TG0h", "WBV"]:
        if feat in X_te.columns:
            shap.plots.scatter(sv[:, feat], color=sv, show=False)
            plt.tight_layout(); plt.savefig(OUTDIR/f"shap_perm_depend_{feat}.png", dpi=300); plt.close()
    print("[INFO] Saved permutation-SHAP plots.")
except Exception as e:
    print("[WARN] Permutation SHAP failed:", repr(e))


# ------------ Ablation: TG-only vs TG+WBV/HCT/TP ------------
X_tg = X[["TG0h","TG4h"]]
X_tg_plus = X[["TG0h","TG4h","WBV","Hematocrit","TotalProtein"]]
def quick_auc(Xm):
    Xtr,Xva,ytr,yva = train_test_split(Xm, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    clf = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                            n_estimators=400, max_depth=3, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=SEED)
    clf.fit(Xtr,ytr); proba = clf.predict_proba(Xva)[:,1]
    return roc_auc_score(yva, proba), average_precision_score(yva, proba)
auc_tg = quick_auc(X_tg); auc_tgplus = quick_auc(X_tg_plus)
with open(OUTDIR/"ablation.txt","w") as f:
    f.write(f"TG-only AUC={auc_tg[0]:.3f}, PR-AUC={auc_tg[1]:.3f}\n")
    f.write(f"TG+WBV/HCT/TP AUC={auc_tgplus[0]:.3f}, PR-AUC={auc_tgplus[1]:.3f}\n")

# ------------ Simple risk score (Logistic) ------------
lr_final = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_tr); Xtr_s = scaler.transform(X_tr); Xte_s = scaler.transform(X_te)
lr_final.fit(Xtr_s, y_tr)
coefs = pd.Series(lr_final.coef_[0], index=X.columns)
coefs.to_csv(OUTDIR/"logistic_coefficients.csv")
# แปลงเป็นคะแนน 0–100
weights = (coefs/np.sum(np.abs(coefs))).fillna(0)
risk_points = (weights.abs()/weights.abs().max()*100).round().astype(int)
risk_points.sort_values(ascending=False).to_csv(OUTDIR/"risk_points_0_100.csv")
print("[DONE] Results saved to", OUTDIR.resolve())
