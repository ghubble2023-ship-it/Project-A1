"""
Retrain the Eyes of Greg photo classifier on the uploaded corpus.

Differences from engine/learning/train_classifier.py (deliberate, and
they only make the numbers HARDER to fake):
  - stratified held-out test split (20%) -> the reported accuracy is on
    photos the model never saw
  - 5-fold cross-validation on the training half -> tells us if the
    accuracy is stable or a fluke of one lucky split
  - StandardScaler before logistic regression -> feature weights become
    comparable to each other, so "which feature mattered" means
    something instead of just reflecting each feature's units
  - a permutation / label-shuffle sanity run -> if accuracy on shuffled
    labels isn't ~50%, something is leaking and the result is garbage

Writes the fitted pipeline to engine/learning/trained_model.joblib and a
human-readable report to /app/ml/train_report.json
"""
import json
import os
import sys

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             classification_report)

REF = "/app/reference_pipeline"
sys.path.insert(0, REF)
from engine.learning.features import FEATURE_NAMES  # noqa: E402

FEATURES_JSON = "/app/ml/features.json"
MODEL_PATH = os.path.join(REF, "engine/learning/trained_model.joblib")
REPORT_PATH = "/app/ml/train_report.json"
SEED = 42


def load():
    with open(FEATURES_JSON) as f:
        rows = json.load(f)
    X, y, paths = [], [], []
    for r in rows:
        feats = r["features"]
        X.append([feats.get(n) for n in FEATURE_NAMES])
        y.append(1 if r["label"] == "fake" else 0)
        paths.append(r["image_path"])
    return np.array(X, dtype=float), np.array(y), paths, rows


def make_pipeline():
    return Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])


def main():
    X, y, paths, rows = load()
    report = {"n_total": int(len(y)), "n_fake": int(y.sum()), "n_real": int((y == 0).sum())}
    print(f"Loaded {len(y)} examples ({y.sum()} fake / {(y==0).sum()} real)")

    # --- coverage / missingness, per class (leak check #1) ---
    cov = {}
    for i, name in enumerate(FEATURE_NAMES):
        col = X[:, i]
        cov[name] = {
            "available_fake": int(np.sum(~np.isnan(col[y == 1]))),
            "available_real": int(np.sum(~np.isnan(col[y == 0]))),
        }
    report["coverage"] = cov
    print("\n--- feature availability (fake / real) ---")
    for n, c in cov.items():
        print(f"  {n:<26} {c['available_fake']:>4} / {c['available_real']:>4}")

    # --- split ---
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    pipe = make_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_acc = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="accuracy")
    cv_auc = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc")
    print(f"\n5-fold CV on training half: acc={cv_acc.mean():.3f} "
          f"(+/-{cv_acc.std():.3f})  auc={cv_auc.mean():.3f}")

    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(yte, pred)
    auc = roc_auc_score(yte, proba)
    cm = confusion_matrix(yte, pred).tolist()
    print(f"HELD-OUT TEST ({len(yte)} photos): acc={acc:.3f}  auc={auc:.3f}")
    print("confusion matrix [[real->real, real->fake], [fake->real, fake->fake]]:")
    print(f"  {cm}")
    print(classification_report(yte, pred, target_names=["real", "fake"]))

    # --- label-shuffle sanity run (leak check #2) ---
    rng = np.random.default_rng(SEED)
    y_shuf = rng.permutation(y)
    shuf_acc = cross_val_score(make_pipeline(), X, y_shuf, cv=cv, scoring="accuracy").mean()
    print(f"label-shuffle control accuracy (should be ~0.500): {shuf_acc:.3f}")

    # --- refit on ALL data for the shipped model ---
    final = make_pipeline()
    final.fit(X, y)
    joblib.dump(final, MODEL_PATH)
    print(f"\nsaved model -> {MODEL_PATH}")

    coefs = final.named_steps["clf"].coef_[0]
    importance = sorted(zip(FEATURE_NAMES, coefs), key=lambda t: abs(t[1]), reverse=True)
    print("\n--- feature weights (standardized, + = pushes toward FAKE) ---")
    for n, w in importance:
        print(f"  {n:<26} {w:+.3f}")

    report.update({
        "cv_accuracy_mean": round(float(cv_acc.mean()), 4),
        "cv_accuracy_std": round(float(cv_acc.std()), 4),
        "cv_auc_mean": round(float(cv_auc.mean()), 4),
        "holdout_n": int(len(yte)),
        "holdout_accuracy": round(float(acc), 4),
        "holdout_auc": round(float(auc), 4),
        "confusion_matrix": cm,
        "label_shuffle_control_accuracy": round(float(shuf_acc), 4),
        "feature_weights": [{"feature": n, "weight": round(float(w), 4)} for n, w in importance],
        "imputer_means": {n: (None if np.isnan(v) else round(float(v), 4))
                          for n, v in zip(FEATURE_NAMES, final.named_steps["impute"].statistics_)},
        "seed": SEED,
    })
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
