"""Scam-type classifier: char + word TF-IDF into logistic regression.

Trained on two public CSV datasets plus a small hand-written seed for `job_task`, which has no
public source. Train on the VM so the pickle is big-endian: `python -m forest.intel.classify`.
"""

import csv
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from forest.config import DATA_DIR

RAW = DATA_DIR / "raw"
SEED = Path(__file__).resolve().parent / "seed" / "job_task.csv"
MODEL_PATH = DATA_DIR / "artifacts" / "scam_clf.joblib"

SHADE_LABELS = {
    "0": "legit",
    "1": "digital_arrest",
    "2": "lottery",
    "3": "kyc_bank",
    "4": "courier_gift",
}
HINGLISH_LABELS = {
    "none": "legit",
    "police_digital_arrest": "digital_arrest",
    "police_blackmail": "digital_arrest",
    "aadhaar": "digital_arrest",
    "bank_kyc": "kyc_bank",
    "amazon": "courier_gift",
    "lottery": "lottery",
    "relative": "family_emergency",
}
SPLIT_SEED = 42
GROUP_WORDS = 6

Rows = list[tuple[str, str]]


def normalise(text: str) -> str:
    """Lower-case, digits and punctuation out: two messages that differ only in an amount or
    an account number collapse to the same string."""
    return re.sub(r"\s+", " ", re.sub(r"[\d\W_]+", " ", text.lower())).strip()


def group_key(text: str) -> str:
    """Messages sharing their opening words are near-duplicates of one template; the split
    keeps each such group on one side."""
    return " ".join(normalise(text).split()[:GROUP_WORDS])


def dedupe(rows: Rows) -> tuple[Rows, int]:
    """Drop repeats of the same normalised text, and drop the text altogether when its copies
    disagree on the label. Returns the rows and how many label-conflict groups were removed."""
    labels: dict[str, set[str]] = defaultdict(set)
    first: dict[str, tuple[str, str]] = {}
    for text, label in rows:
        key = normalise(text)
        labels[key].add(label)
        first.setdefault(key, (text, label))
    conflicts = sum(len(v) > 1 for v in labels.values())
    return [r for key, r in first.items() if len(labels[key]) == 1], conflicts


def read_csv(path: Path, label_col: str, mapping: dict[str, str] | None) -> Rows:
    with open(path, newline="", encoding="utf-8") as f:
        raw = [(r["text"], r[label_col]) for r in csv.DictReader(f)]
    return [(t, mapping[label] if mapping else label) for t, label in raw]


def load_sources() -> dict[str, tuple[Rows, dict]]:
    """Source name -> (deduplicated rows, dedupe stats)."""
    spec = {
        "shade63": (RAW / "scam_multiclass.csv", "label", SHADE_LABELS),
        "hinglish_calls": (RAW / "scam_calls_hinglish.csv", "scam_category", HINGLISH_LABELS),
        "job_task_seed": (SEED, "label", None),
    }
    out = {}
    for name, (path, col, mapping) in spec.items():
        rows = read_csv(path, col, mapping)
        kept, conflicts = dedupe(rows)
        out[name] = (kept, {"raw": len(rows), "kept": len(kept), "label_conflicts": conflicts})
    return out


def build_model() -> Pipeline:
    features = FeatureUnion(
        [
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True
                ),
            ),
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ]
    )
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    return Pipeline([("tfidf", features), ("lr", clf)])


def fit(rows: Rows) -> Pipeline:
    texts, labels = zip(*rows, strict=True)
    return build_model().fit(list(texts), list(labels))


def train_and_save(path: Path = MODEL_PATH) -> Pipeline:
    rows = [r for rows, _ in load_sources().values() for r in rows]
    model = fit(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return model


@lru_cache(maxsize=1)
def _model() -> Pipeline:
    return joblib.load(MODEL_PATH)


def proba(texts: list[str]) -> np.ndarray:
    """Class probabilities, columns in `classes()` order."""
    return _model().predict_proba(texts)


def classes() -> list[str]:
    return list(_model().classes_)


def classify(text: str) -> tuple[str, float]:
    """Most likely scam type for one message and its probability."""
    p = proba([text])[0]
    i = int(p.argmax())
    return classes()[i], float(p[i])


def classify_conversation(texts: list[str]) -> tuple[str, float]:
    """Label for a whole chat: argmax of the summed per-message probabilities, reported as that
    class's share of the total."""
    total = proba(texts).sum(axis=0)
    i = int(total.argmax())
    return classes()[i], float(total[i] / total.sum())


if __name__ == "__main__":
    train_and_save()
    print(f"saved {MODEL_PATH}")
