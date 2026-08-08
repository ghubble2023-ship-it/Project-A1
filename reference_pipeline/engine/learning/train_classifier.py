"""
Trainable classifier - learns weights from labeled examples instead of
me hand-picking thresholds like "delta > 8 degrees."

Uses logistic regression over the real feature set in features.py, with
mean-imputation for missing values (a photo where eye detection failed
doesn't get thrown away - it gets the average value for that feature,
which is the standard, honest way to handle missing data, not a hack).

HONEST GUARDRAILS built in, not just mentioned in a README:
  - Refuses to train with fewer than MIN_EXAMPLES_PER_CLASS examples
    per class - returns a clear error instead of silently producing a
    model that just memorized 2 photos.
  - Reports which features actually mattered (coefficients), so you
    can see if it's learning something sane or something spurious.
"""

import os
import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .features import extract_features, features_to_vector, FEATURE_NAMES
from .labeled_dataset import LabeledDataset

MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_model.joblib")
MIN_EXAMPLES_PER_CLASS = 8  # below this, refuse to train - see docstring


def train(dataset: LabeledDataset):
    examples = dataset.all()
    counts = dataset.count_by_label()

    if counts["fake"] < MIN_EXAMPLES_PER_CLASS or counts["real"] < MIN_EXAMPLES_PER_CLASS:
        return {
            "trained": False,
            "reason": (
                f"Need at least {MIN_EXAMPLES_PER_CLASS} labeled examples of EACH "
                f"class to train something that isn't just memorizing. "
                f"Currently have {counts['fake']} fake, {counts['real']} real. "
                f"This isn't a made-up limit - fewer than this and the model "
                f"can't do anything but memorize, which means it will look "
                f"perfect on your own examples and fail on the next new photo."
            ),
            "counts": counts,
        }

    X, y = [], []
    for ex in examples:
        feats = extract_features(ex["image_path"])
        X.append(features_to_vector(feats))
        y.append(1 if ex["label"] == "fake" else 0)

    X = np.array(X, dtype=float)
    y = np.array(y)

    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(X, y)

    joblib.dump(pipeline, MODEL_PATH)

    coefs = pipeline.named_steps["clf"].coef_[0]
    feature_importance = sorted(
        zip(FEATURE_NAMES, coefs), key=lambda t: abs(t[1]), reverse=True
    )

    return {
        "trained": True,
        "n_examples": len(examples),
        "counts": counts,
        "feature_importance": [{"feature": f, "weight": round(float(w), 3)} for f, w in feature_importance],
    }


def predict(image_path: str):
    if not os.path.exists(MODEL_PATH):
        return {"error": "No trained model yet - call train() with enough labeled examples first."}

    pipeline = joblib.load(MODEL_PATH)
    feats = extract_features(image_path)
    vec = np.array([features_to_vector(feats)], dtype=float)
    proba_fake = float(pipeline.predict_proba(vec)[0][1])

    return {
        "probability_fake": round(proba_fake, 3),
        "features_used": feats,
    }
