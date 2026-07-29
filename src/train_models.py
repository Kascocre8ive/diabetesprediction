"""
Step 2 of the pipeline: train the three base models (SVM, Logistic
Regression, XGBoost), combine them into a soft-voting hybrid ensemble,
evaluate everything, and save the trained artifacts for the Streamlit app.

Run directly (after src/preprocessing.py has produced models/splits.pkl):
    python -m src.train_models

NOTE on runtime: SVM with an RBF kernel scales badly (~O(n^2)-O(n^3)) and
on ~85,000+ balanced training rows a full grid search can take a very long
time (the project's own literature review — Fadli Kurniawan & Megawaty,
2025 — reports one SVM run taking ~3500 seconds). To keep this runnable on
a student laptop we tune/train SVM on a stratified subsample
(SVM_TRAIN_SAMPLE_SIZE in this file). Increase or remove the subsample if
you have more time/compute; the option is called out clearly so you can
justify it in Chapter 4.
"""
import sys
import os
import json
import time
import joblib
import numpy as np
from scipy import stats
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, confusion_matrix,
)
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.ensemble import SoftVotingEnsemble

# Subsample size used only for SVM fitting/tuning (see module docstring).
# Set to None to train SVM on the full balanced training set.
SVM_TRAIN_SAMPLE_SIZE = 15000
N_ITER_SEARCH = 8          # RandomizedSearchCV budget per model
CV_FOLDS = 3


def load_splits():
    path = os.path.join(config.MODEL_DIR, "splits.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "models/splits.pkl not found. Run `python -m src.preprocessing` first."
        )
    return joblib.load(path)


def subsample(X, y, n, seed=config.RANDOM_STATE):
    if n is None or n >= len(y):
        return X, y
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n_each = n // 2
    idx = np.concatenate([
        rng.choice(idx_pos, size=min(n_each, len(idx_pos)), replace=False),
        rng.choice(idx_neg, size=min(n_each, len(idx_neg)), replace=False),
    ])
    rng.shuffle(idx)
    return X[idx], y[idx]


def train_svm(X_train, y_train):
    print("\n[1/3] Tuning SVM...")
    X_sub, y_sub = subsample(X_train, y_train, SVM_TRAIN_SAMPLE_SIZE)
    print(f"  training on {len(y_sub):,} rows "
          f"({'full set' if SVM_TRAIN_SAMPLE_SIZE is None else 'subsample'})")

    param_dist = {
        "C": [0.1, 1, 10, 100],
        "kernel": ["rbf", "linear", "poly"],
        "gamma": ["scale", "auto"],
    }
    search = RandomizedSearchCV(
        SVC(probability=True, random_state=config.RANDOM_STATE),
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,
        cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE),
        scoring="f1",
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )
    t0 = time.time()
    search.fit(X_sub, y_sub)
    print(f"  best params: {search.best_params_}  ({time.time() - t0:.1f}s)")
    return search.best_estimator_


def train_logistic_regression(X_train, y_train):
    print("\n[2/3] Tuning Logistic Regression...")
    param_dist = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
    }
    search = RandomizedSearchCV(
        LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE),
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,
        cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE),
        scoring="f1",
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
        error_score=0.0,
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    print(f"  best params: {search.best_params_}  ({time.time() - t0:.1f}s)")
    return search.best_estimator_


def train_xgboost(X_train, y_train):
    print("\n[3/3] Tuning XGBoost...")
    param_dist = {
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 6, 9],
        "n_estimators": [100, 200, 500],
        "subsample": [0.8, 0.9, 1.0],
    }
    search = RandomizedSearchCV(
        XGBClassifier(
            eval_metric="logloss",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,
        cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE),
        scoring="f1",
        random_state=config.RANDOM_STATE,
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    print(f"  best params: {search.best_params_}  ({time.time() - t0:.1f}s)")
    return search.best_estimator_


def tune_ensemble_weights(models, names, X_val, y_val):
    """Small grid search over (SVM, LR, XGBoost) weight combinations,
    selected using the held-out validation split (never the test set)."""
    print("\nTuning ensemble weights on validation split...")
    candidate_weights = [
        (1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 1, 1),
        (1, 1, 3), (1, 3, 1), (3, 1, 1), (2, 1, 2), (1, 2, 2),
    ]
    best_f1, best_w = -1, (1, 1, 1)
    for w in candidate_weights:
        ens = SoftVotingEnsemble(models=models, weights=list(w), names=names)
        preds = ens.predict(X_val)
        f1 = f1_score(y_val, preds)
        if f1 > best_f1:
            best_f1, best_w = f1, w
    print(f"  best weights (SVM, LR, XGBoost) = {best_w}  (val F1={best_f1:.4f})")
    return list(best_w)


def evaluate(name, y_true, y_pred, y_proba):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
        "auc_roc": round(roc_auc_score(y_true, y_proba), 4),
    }


def mcnemar_test(y_true, pred_a, pred_b):
    """Manual McNemar's test (exact binomial for small counts, else
    chi-square with continuity correction) between two models' predictions."""
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    b = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))   # A wrong, B right
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}
    if n < 25:
        p = 2 * stats.binom.cdf(min(b, c), n, 0.5)
        p = min(p, 1.0)
        stat = None
    else:
        stat = (abs(b - c) - 1) ** 2 / n
        p = 1 - stats.chi2.cdf(stat, df=1)
    return {"b": b, "c": c, "statistic": stat, "p_value": round(float(p), 6)}


def build_eval_artifacts(name, y_true, y_pred, y_proba):
    """Confusion matrix + ROC/PR curve points for one model, JSON/joblib-friendly."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "confusion_matrix": cm.tolist(),
        "roc_fpr": fpr.tolist(),
        "roc_tpr": tpr.tolist(),
        "pr_precision": prec.tolist(),
        "pr_recall": rec.tolist(),
    }


def run():
    data = load_splits()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    svm_model = train_svm(X_train, y_train)
    lr_model = train_logistic_regression(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    models = [svm_model, lr_model, xgb_model]
    names = ["SVM", "Logistic Regression", "XGBoost"]

    weights = tune_ensemble_weights(models, names, X_val, y_val)
    ensemble = SoftVotingEnsemble(models=models, weights=weights, names=names)

    # --- Final evaluation on the held-out test set (touched once) ---
    print("\nEvaluating on the held-out test set...")
    metrics = {}
    preds_by_model = {}
    probas_by_model = {}
    eval_artifacts = {"models": {}}

    for name, model in zip(names, models):
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        preds_by_model[name] = pred
        probas_by_model[name] = proba
        metrics[name] = evaluate(name, y_test, pred, proba)
        eval_artifacts["models"][name] = build_eval_artifacts(name, y_test, pred, proba)
        print(f"  {name}: {metrics[name]}")

    ens_proba = ensemble.predict_proba(X_test)[:, 1]
    ens_pred = ensemble.predict(X_test)
    preds_by_model["Hybrid Ensemble"] = ens_pred
    probas_by_model["Hybrid Ensemble"] = ens_proba
    metrics["Hybrid Ensemble"] = evaluate("Hybrid Ensemble", y_test, ens_pred, ens_proba)
    eval_artifacts["models"]["Hybrid Ensemble"] = build_eval_artifacts(
        "Hybrid Ensemble", y_test, ens_pred, ens_proba
    )
    print(f"  Hybrid Ensemble: {metrics['Hybrid Ensemble']}")

    # --- McNemar's test: ensemble vs each base model ---
    print("\nMcNemar's test (Hybrid Ensemble vs each base model):")
    mcnemar_results = {}
    for name in names:
        res = mcnemar_test(y_test, ens_pred, preds_by_model[name])
        mcnemar_results[f"Hybrid Ensemble vs {name}"] = res
        sig = "significant (p<0.05)" if res["p_value"] < 0.05 else "not significant"
        print(f"  vs {name}: p={res['p_value']}  -> {sig}")

    # --- Save everything ---
    joblib.dump(svm_model, config.SVM_MODEL_PATH)
    joblib.dump(lr_model, config.LR_MODEL_PATH)
    joblib.dump(xgb_model, config.XGB_MODEL_PATH)
    joblib.dump(ensemble, config.ENSEMBLE_MODEL_PATH)
    joblib.dump(eval_artifacts, config.EVAL_ARTIFACTS_PATH)

    with open(config.METRICS_PATH, "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "mcnemar": mcnemar_results,
                "ensemble_weights": dict(zip(names, weights)),
            },
            f,
            indent=2,
        )

    feature_importance = dict(
        zip(data["feature_names"], [round(float(v), 4) for v in xgb_model.feature_importances_])
    )
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)
    )
    with open(config.FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump(feature_importance, f, indent=2)

    print(f"\nSaved models to {config.MODEL_DIR}/")
    print(f"Saved metrics to {config.METRICS_PATH}")
    print(f"Saved feature importance to {config.FEATURE_IMPORTANCE_PATH}")
    print(f"Saved evaluation artifacts to {config.EVAL_ARTIFACTS_PATH}")


if __name__ == "__main__":
    run()