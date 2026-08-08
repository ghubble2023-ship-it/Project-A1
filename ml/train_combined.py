"""
Combined model: the original 8 forensic features + the spectral /
residual block. Same honesty rules as train_model.py - stratified
held-out test, 5-fold CV, label-shuffle control.

Also reports each block on its own, so we can see exactly where the
accuracy comes from instead of just claiming a big number.
"""
import json
import sys

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             classification_report)

sys.path.insert(0, "/app/reference_pipeline")
sys.path.insert(0, "/app/ml")
from engine.learning.features import FEATURE_NAMES as BASE_NAMES  # noqa: E402
import spectral_features as sf  # noqa: E402

SPEC_NAMES = sf.FEATURE_NAMES
ALL_NAMES = list(BASE_NAMES) + list(SPEC_NAMES)
SEED = 42
MODEL_OUT = "/app/ml/photo_model.joblib"
META_OUT = "/app/ml/photo_model_meta.json"
REPORT_OUT = "/app/ml/train_report_v2.json"


def load():
    base = {r["image_path"]: r for r in json.load(open("/app/ml/features.json"))}
    spec = {r["image_path"]: r for r in json.load(open("/app/ml/features_spectral.json"))}
    paths = sorted(set(base) & set(spec))
    X, y = [], []
    for p in paths:
        row = [base[p]["features"].get(n) for n in BASE_NAMES]
        row += [spec[p]["features"].get(n) for n in SPEC_NAMES]
        X.append(row)
        y.append(1 if base[p]["label"] == "fake" else 0)
    return np.array(X, dtype=float), np.array(y), paths


def logreg():
    return Pipeline([("impute", SimpleImputer(strategy="mean")),
                     ("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=5000, C=1.0))])


def gbm():
    return Pipeline([("impute", SimpleImputer(strategy="mean")),
                     ("clf", GradientBoostingClassifier(random_state=SEED))])


def evaluate(name, make, X, y, report):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cv_acc = cross_val_score(make(), Xtr, ytr, cv=cv, scoring="accuracy")
    cv_auc = cross_val_score(make(), Xtr, ytr, cv=cv, scoring="roc_auc")
    m = make()
    m.fit(Xtr, ytr)
    proba = m.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc, auc = accuracy_score(yte, pred), roc_auc_score(yte, proba)
    print(f"\n=== {name} ===")
    print(f"  5-fold CV (train half): acc={cv_acc.mean():.3f} (+/-{cv_acc.std():.3f}) auc={cv_auc.mean():.3f}")
    print(f"  HELD-OUT ({len(yte)}):      acc={acc:.3f} auc={auc:.3f}")
    print(f"  confusion [[real->real, real->fake],[fake->real, fake->fake]] = {confusion_matrix(yte, pred).tolist()}")
    print(classification_report(yte, pred, target_names=["real", "fake"], digits=3))
    report[name] = {
        "cv_accuracy_mean": round(float(cv_acc.mean()), 4),
        "cv_accuracy_std": round(float(cv_acc.std()), 4),
        "cv_auc_mean": round(float(cv_auc.mean()), 4),
        "holdout_accuracy": round(float(acc), 4),
        "holdout_auc": round(float(auc), 4),
        "confusion_matrix": confusion_matrix(yte, pred).tolist(),
    }
    return acc


def main():
    X, y, paths = load()
    print(f"{len(y)} examples ({int(y.sum())} fake / {int((y==0).sum())} real), "
          f"{X.shape[1]} features")
    report = {"n_total": int(len(y)), "feature_names": ALL_NAMES}

    nb = len(BASE_NAMES)
    idx_base = list(range(nb))
    idx_spec = list(range(nb, X.shape[1]))

    evaluate("base8_logreg", logreg, X[:, idx_base], y, report)
    evaluate("spectral_only_logreg", logreg, X[:, idx_spec], y, report)
    evaluate("combined_logreg", logreg, X, y, report)
    evaluate("combined_gbm", gbm, X, y, report)

    # label-shuffle control on the full combined set
    rng = np.random.default_rng(SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    shuf = cross_val_score(logreg(), X, rng.permutation(y), cv=cv, scoring="accuracy").mean()
    print(f"\nlabel-shuffle control (should be ~0.500): {shuf:.3f}")
    report["label_shuffle_control_accuracy"] = round(float(shuf), 4)

    # ship the combined logistic-regression model (interpretable weights)
    final = logreg()
    final.fit(X, y)
    joblib.dump(final, MODEL_OUT)
    coefs = final.named_steps["clf"].coef_[0]
    kept = [n for i, n in enumerate(ALL_NAMES)
            if not np.all(np.isnan(X[:, i]))]
    importance = sorted(zip(kept, coefs), key=lambda t: abs(t[1]), reverse=True)
    print("\n--- top 12 weights (standardized, + = pushes toward FAKE) ---")
    for n, w in importance[:12]:
        print(f"  {n:<26} {w:+.3f}")

    means = final.named_steps["impute"].statistics_
    json.dump({
        "feature_names": ALL_NAMES,
        "base_feature_names": list(BASE_NAMES),
        "spectral_feature_names": list(SPEC_NAMES),
        "kept_feature_names": kept,
        "imputer_means": {n: (None if np.isnan(v) else float(v)) for n, v in zip(kept, means)},
        "trained_on": {"fake": int(y.sum()), "real": int((y == 0).sum())},
        "holdout_accuracy": report["combined_logreg"]["holdout_accuracy"],
        "holdout_auc": report["combined_logreg"]["holdout_auc"],
        "cv_accuracy_mean": report["combined_logreg"]["cv_accuracy_mean"],
        "feature_weights": [{"feature": n, "weight": round(float(w), 4)} for n, w in importance],
    }, open(META_OUT, "w"), indent=2)
    report["feature_weights"] = [{"feature": n, "weight": round(float(w), 4)} for n, w in importance]
    json.dump(report, open(REPORT_OUT, "w"), indent=2)
    print(f"\nmodel -> {MODEL_OUT}\nmeta  -> {META_OUT}\nreport -> {REPORT_OUT}")


if __name__ == "__main__":
    main()
