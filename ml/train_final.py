"""
Final photo model: SRM-lite co-occurrence + spectral block + the original
8 forensic features, with a regularisation sweep chosen by
cross-validation on the training half ONLY (the held-out fifth is never
used to pick anything).
"""
import json
import sys

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)

sys.path.insert(0, "/app/reference_pipeline")
sys.path.insert(0, "/app/ml")
from engine.learning.features import FEATURE_NAMES as BASE_NAMES  # noqa: E402
import spectral_features as sfmod  # noqa: E402
import srm_features as srm  # noqa: E402

SEED = 42
SPEC_NAMES = list(sfmod.FEATURE_NAMES)
SRM_NAMES = [f"srm_{i}" for i in range(srm.DIM)]
ALL_NAMES = list(BASE_NAMES) + SPEC_NAMES + SRM_NAMES

MODEL_OUT = "/app/ml/photo_model.joblib"
META_OUT = "/app/ml/photo_model_meta.json"
REPORT_OUT = "/app/ml/train_report_final.json"


def load():
    base = {r["image_path"]: r for r in json.load(open("/app/ml/features.json"))}
    spec = {r["image_path"]: r for r in json.load(open("/app/ml/features_spectral.json"))}
    d = np.load("/app/ml/srm_cache.npz", allow_pickle=True)
    srm_map = {p: d["X"][i] for i, p in enumerate(d["paths"])}

    paths = sorted(set(base) & set(spec) & set(srm_map))
    X, y = [], []
    for p in paths:
        row = [base[p]["features"].get(n) for n in BASE_NAMES]
        row += [spec[p]["features"].get(n) for n in SPEC_NAMES]
        X.append(np.concatenate([np.array(row, dtype=float), srm_map[p]]))
        y.append(1 if base[p]["label"] == "fake" else 0)
    return np.stack(X), np.array(y), paths


def pipe(C):
    return Pipeline([("impute", SimpleImputer(strategy="mean")),
                     ("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=5000, C=C))])


def main():
    X, y, paths = load()
    print(f"X={X.shape}  fake={int(y.sum())} real={int((y==0).sum())}")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    results = {}
    for C in [0.001, 0.003, 0.01, 0.03, 0.1]:
        s = cross_val_score(pipe(C), Xtr, ytr, cv=cv, scoring="accuracy", n_jobs=5)
        results[C] = (s.mean(), s.std())
        print(f"  C={C:<6} CV acc={s.mean():.3f} (+/-{s.std():.3f})")
    bestC = max(results, key=lambda c: results[c][0])
    print(f"chosen C={bestC} (by CV on training half only)")

    m = pipe(bestC)
    m.fit(Xtr, ytr)
    proba = m.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc, auc = accuracy_score(yte, pred), roc_auc_score(yte, proba)
    cm = confusion_matrix(yte, pred).tolist()
    print(f"\nHELD-OUT ({len(yte)} photos never seen or used for tuning):")
    print(f"  accuracy={acc:.3f}  auc={auc:.3f}")
    print(f"  confusion [[real->real, real->fake],[fake->real, fake->fake]] = {cm}")
    print(classification_report(yte, pred, target_names=["real", "fake"], digits=3))

    # threshold that keeps false accusations of real people low (<=10% FPR)
    fpr, tpr, thr = roc_curve(yte, proba)
    ok = np.where(fpr <= 0.10)[0]
    t_cons = float(thr[ok[-1]]) if len(ok) else 0.5
    print(f"conservative threshold for <=10% false-accusation rate: {t_cons:.3f} "
          f"(catches {tpr[ok[-1]]*100:.0f}% of fakes at that setting)")

    rng = np.random.default_rng(SEED)
    shuf = cross_val_score(pipe(bestC), X, rng.permutation(y), cv=cv,
                           scoring="accuracy", n_jobs=5).mean()
    print(f"label-shuffle control (should be ~0.500): {shuf:.3f}")

    # refit on everything for the shipped model
    final = pipe(bestC)
    final.fit(X, y)
    joblib.dump(final, MODEL_OUT)

    kept = [n for i, n in enumerate(ALL_NAMES) if not np.all(np.isnan(X[:, i]))]
    coefs = final.named_steps["clf"].coef_[0]
    imp = sorted(zip(kept, coefs), key=lambda t: abs(t[1]), reverse=True)
    named_imp = [(n, w) for n, w in imp if not n.startswith("srm_")][:10]
    print("\n--- weights of the human-readable features (+ = toward FAKE) ---")
    for n, w in named_imp:
        print(f"  {n:<26} {w:+.4f}")
    srm_share = sum(abs(w) for n, w in imp if n.startswith("srm_")) / sum(abs(w) for _, w in imp)
    print(f"\nshare of total model weight carried by the SRM co-occurrence block: {srm_share*100:.1f}%")

    means = final.named_steps["impute"].statistics_
    meta = {
        "model_version": "eog-photo-v2",
        "feature_order": ALL_NAMES,
        "base_feature_names": list(BASE_NAMES),
        "spectral_feature_names": SPEC_NAMES,
        "srm_dim": srm.DIM,
        "kept_feature_names": kept,
        "C": bestC,
        "trained_on": {"fake": int(y.sum()), "real": int((y == 0).sum())},
        "holdout_accuracy": round(float(acc), 4),
        "holdout_auc": round(float(auc), 4),
        "cv_accuracy_mean": round(float(results[bestC][0]), 4),
        "cv_accuracy_std": round(float(results[bestC][1]), 4),
        "label_shuffle_control_accuracy": round(float(shuf), 4),
        "confusion_matrix": cm,
        "conservative_threshold_10pct_fpr": round(t_cons, 4),
        "imputer_means_named": {n: (None if np.isnan(v) else float(v))
                                for n, v in zip(kept, means) if not n.startswith("srm_")},
        "named_feature_weights": [{"feature": n, "weight": round(float(w), 5)} for n, w in imp
                                  if not n.startswith("srm_")],
        "srm_weight_share": round(float(srm_share), 4),
    }
    json.dump(meta, open(META_OUT, "w"), indent=2)
    json.dump({"cv_sweep": {str(k): v for k, v in results.items()}, **meta},
              open(REPORT_OUT, "w"), indent=2)
    print(f"\nmodel -> {MODEL_OUT}\nmeta -> {META_OUT}")


if __name__ == "__main__":
    main()
