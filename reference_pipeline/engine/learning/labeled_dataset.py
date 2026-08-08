"""
Labeled dataset - this is the actual "teaching" part.

Every time Greg (or Grok, or anyone on the project) confirms "this one
was fake" or "this one was real," that becomes a labeled example here.
Over time this file grows into real training data. This is what makes
the system teachable rather than hardcoded - the classifier in
train_classifier.py learns its weights FROM this file, not from my
guesses.

Honest note: right now this has a handful of examples. That is not
enough to train anything that generalizes - it's enough to prove the
pipeline runs end to end. A real classifier needs dozens to hundreds
of labeled examples per class before its weights mean anything. Every
case Greg (or Meta, Grok, Kimi) confirms should get added here.
"""

import json
import os
import time

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "labeled_dataset.json")


class LabeledDataset:
    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                self.examples = json.load(f)
        else:
            self.examples = []

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.examples, f, indent=2)

    def add(self, image_path: str, label: str, notes: str = "", labeled_by: str = "greg"):
        """
        label: "fake" or "real" - Greg's (or another reviewer's) actual
        judgment call. This IS the ground truth the classifier learns
        from - there is no other source of truth here.
        """
        if label not in ("fake", "real"):
            raise ValueError("label must be 'fake' or 'real'")
        entry = {
            "image_path": image_path,
            "label": label,
            "notes": notes,
            "labeled_by": labeled_by,
            "t": time.time(),
        }
        self.examples.append(entry)
        self._save()
        return entry

    def count_by_label(self):
        counts = {"fake": 0, "real": 0}
        for e in self.examples:
            counts[e["label"]] += 1
        return counts

    def all(self):
        return list(self.examples)
