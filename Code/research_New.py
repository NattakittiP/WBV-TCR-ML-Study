import argparse, json, os
else:
groups = np.arange(len(df_ml)) # fallback: each row own group


X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
X, y, groups, test_size=0.2, stratify=y, random_state=42
)


# Preprocess
num_pipe = Pipeline([
('impute', SimpleImputer(strategy='median')),
('scale', StandardScaler())
])
cat_pipe = Pipeline([
('impute', SimpleImputer(strategy='most_frequent')),
('onehot', OneHotEncoder(handle_unknown='ignore'))
])


pre = ColumnTransformer([
('num', num_pipe, NUM_FEATURES),
('cat', cat_pipe, CAT_FEATURES)
])


# Models
logreg = Pipeline([
('pre', pre),
('clf', LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced'))
])


gbc = Pipeline([
('pre', pre),
('clf', GradientBoostingClassifier())
])


# Nested CV (training only)
inner = StratifiedGroupKFold(n_splits=5)
param_lr = {
'clf__C': [0.1, 1.0, 3.0],
}
param_gbc = {
'clf__n_estimators': [150, 300],
'clf__max_depth': [2, 3],
'clf__learning_rate': [0.05, 0.1]
}


cv_lr = GridSearchCV(logreg, param_lr, scoring='roc_auc', cv=inner, n_jobs=-1)
cv_gbc = GridSearchCV(gbc, param_gbc, scoring='roc_auc', cv=inner, n_jobs=-1)


cv_lr.fit(X_train, y_train, groups=g_train)
cv_gbc.fit(X_train, y_train, groups=g_train)


best_lr = cv_lr.best_estimator_
best_gbc = cv_gbc.best_estimator_


# Evaluate on untouched test
probs_lr = best_lr.predict_proba(X_test)[:,1]
probs_gbc = best_gbc.predict_proba(X_test)[:,1]


metrics = {}
for name, p in [('logreg', probs_lr), ('gbc', probs_gbc)]:
metrics[name] = {
'roc_auc': float(roc_auc_score(y_test, p)),
'pr_auc': float(average_precision_score(y_test, p)),
'brier': float(brier_score_loss(y_test, p))
}


# Decision curve example
ths, nb_lr, prev = decision_curve(y_test, probs_lr)
ths, nb_gb, _ = decision_curve(y_test, probs_gbc, thresholds=ths)


out = {
'features_num': NUM_FEATURES,
'features_cat': CAT_FEATURES,
'leak_columns_removed': LEAK_COLUMNS,
'best_params': {
'logreg': cv_lr.best_params_,
'gbc': cv_gbc.best_params_
},
'test_metrics': metrics,
'prevalence_test': float(np.mean(y_test)),
}


print(json.dumps(out, indent=2))