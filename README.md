# 📘 WBV–TCR Machine Learning Pipeline (Leakage-Controlled)
_A fully reproducible clinical ML workflow featuring nested cross-validation, calibration, SHAP explainability, bootstrap confidence intervals, and ablation studies._

---

## 📌 Description
This repository provides a complete and rigorously designed **machine-learning pipeline** for evaluating the relationship between **Whole Blood Viscosity (WBV)** and **Triglyceride Clearance Rate (TCR)**.

The workflow implements:

- **Strict leakage control** across all steps  
- **5×5 nested cross-validation** for unbiased model estimation  
- **Isotonic probability calibration** to improve clinical usability  
- **Permutation-based SHAP explainability**  
- **Bootstrap AUC confidence intervals**  
- **Ablation experiments** comparing TG-only models against TG+WBV/HCT/TP models  

This project demonstrates reproducible, transparent, and clinically interpretable methodology suitable for **biomedical** and **bioinformatics machine-learning** research.

---

## 🚀 Key Features

- **Leakage-controlled ML pipeline** (no preprocessing outside CV folds)  
- **5×5 Nested cross-validation** for robust performance estimation  
- **Models included:** Logistic Regression, Random Forest, XGBoost  
- **Isotonic calibration** for probability refinement  
- **Permutation-SHAP explainability** (reproducible & model-agnostic)  
- **Ablation study:** TG-only vs. TG+WBV/HCT/TP  
- **Bootstrap 95% confidence intervals** for ROC-AUC  
- **ROC, PR, and calibration curves**  
- **Permutation-based feature importance (RF)**  
- **Automated risk-score generation** based on logistic regression coefficients  

---

## 🧬 About the Dataset
The dataset used in this study is a **fully anonymized secondary clinical dataset** obtained under institutional data-use restrictions.

> **Note:**  
> Raw clinical data **cannot be distributed** in this repository due to privacy constraints.  
> All scripts will run successfully once the user provides a dataset with identical column names.

---

## 🛠 Methodology Overview

### **1. Data Preparation**
- Forced numeric conversion  
- Median imputation for missing values  
- Sex recoding (M/F → 1/0)  
- Construction of a binary target using TCR quartiles (Q1 cutoff) when necessary  

---

### **2. Hold-Out Split**
- **80/20 stratified train/test split**  
- All preprocessing performed **inside** each CV fold to prevent leakage  

---

### **3. Nested Cross-Validation**
- **Outer loop:** 5 folds  
- **Inner loop:** 5 folds  
- **Randomized hyperparameter search**  
- Fully leakage-controlled preprocessing inside pipelines  

---

### **4. Model Calibration**
- **Isotonic-calibrated Logistic Regression, Random Forest, and XGBoost**  
- Test-set evaluation using **Youden J-optimized thresholds**  

---

### **5. Explainability**
- **Permutation-based SHAP values**  
- Global beeswarm visualization  
- Dependence plots for TG0h, TG4h, WBV  
- Permutation feature importance for Random Forest  

---

### **6. Ablation Studies**
- TG0h/TG4h only  
- TG0h/TG4h + WBV/HCT/TP  
- Comparison of ROC-AUC and PR-AUC across feature sets  

---

### **7. Statistical Rigor**
- Bootstrap **95% confidence intervals** for ROC-AUC  
- Comprehensive metrics:  
  - AUC  
  - PR-AUC  
  - F1-score  
  - Accuracy  
  - Brier Score  
  - Confusion Matrix  

