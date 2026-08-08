"""
Opener corpus - a separate collection from the photo dataset, for the
TEXT of how scam conversations begin.

Greg's real, experienced point: a generic first message ("Hi, this is
[name], how are you today?") triggers none of text_engine's existing
flags - and that's not an oversight, it's the point. Scam openers are
optimized to be unremarkable. The signal isn't in any one opener's
words; it's in whether many different scam contacts open with
near-identical phrasing. That's a real, testable hypothesis - this
module is what tests it.

Needs volume before it can show anything, same as every other part of
this project - 1 example proves nothing, same honesty rule as the
photo dataset.

PRIVACY SCOPE (flagged during review): this corpus stores the account
handles of suspected scam profiles in its context field. That is fine
for Greg's own local evidence log - it is his research data about
accounts that contacted him - but it means this module is a
DEV/RESEARCH TOOL, not shippable as-is. A shipped build must not
persist other people's handles on a user's device. If opener matching
ships, it should carry only the derived move-patterns (see
opener_matcher.py), never the source handles or raw texts.
"""

import json
import os
import time
from itertools import combinations

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "opener_corpus.json")
MIN_EXAMPLES_FOR_CLUSTERING = 8  # same honesty bar as the photo classifier


class OpenerCorpus:
    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                self.entries = json.load(f)
        else:
            self.entries = []

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.entries, f, indent=2)

    def add(self, text: str, context: dict = None, label: str = "suspected_scam"):
        """
        context: e.g. {"unknown_number": True, "had_photo_attached": True,
                       "platform": "SMS"}
        label: "suspected_scam", "confirmed_scam", or "legitimate" -
        keep these honestly distinct, don't collapse suspicion into
        confirmation.
        """
        entry = {
            "text": text,
            "context": context or {},
            "label": label,
            "t": time.time(),
        }
        self.entries.append(entry)
        self._save()
        return entry

    def all(self):
        return list(self.entries)

    def count(self):
        return len(self.entries)


def cluster_by_similarity(corpus: OpenerCorpus, similarity_threshold: float = 0.7):
    """
    Real text-similarity clustering using TF-IDF + cosine similarity
    (scikit-learn - same library already used for the photo
    classifier). Groups openers that are near-duplicates of each
    other, which is exactly what would show a common template.

    Refuses to produce anything below MIN_EXAMPLES_FOR_CLUSTERING,
    same honesty guardrail as train_classifier.py - a handful of
    examples can't show a "common template," they can only show
    themselves.
    """
    texts = [e["text"] for e in corpus.all()]

    if len(texts) < MIN_EXAMPLES_FOR_CLUSTERING:
        return {
            "clustered": False,
            "reason": (
                f"Need at least {MIN_EXAMPLES_FOR_CLUSTERING} opener texts to look "
                f"for a common template - currently have {len(texts)}. Fewer than "
                f"this and any 'match' is just coincidence, not a template."
            ),
            "count": len(texts),
        }

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer().fit_transform(texts)
    sim_matrix = cosine_similarity(vectorizer)

    # Union-find style grouping: connect any two texts above threshold
    n = len(texts)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    pairs_checked = []
    for i, j in combinations(range(n), 2):
        sim = float(sim_matrix[i, j])
        pairs_checked.append({"a": texts[i], "b": texts[j], "similarity": round(sim, 3)})
        if sim >= similarity_threshold:
            union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(texts[i])

    clusters = [g for g in groups.values() if len(g) > 1]
    singletons = [g[0] for g in groups.values() if len(g) == 1]

    return {
        "clustered": True,
        "n_examples": n,
        "n_clusters_found": len(clusters),
        "clusters": clusters,
        "unclustered_singletons": singletons,
        "note": (
            "A cluster here means those opener texts are near-duplicates "
            "of each other by wording - real evidence of a shared template, "
            "not a guess."
        ),
    }
