"""The classifier: character n-gram TF-IDF + logistic regression.

Steps: scrub contact details -> normalize -> cut long comments into windows ->
TF-IDF of character n-grams -> logistic regression -> highest window score ->
sigmoid -> label 1 if the probability is at least the threshold.
"""
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from . import pii
from .normalize import normalize


@dataclass
class Settings:
    analyzer: str = "char_wb"          # char_wb keeps n-grams inside one word, char also crosses spaces
    ngram_range: tuple = (2, 5)
    fold: bool = False                 # write Kazakh letters as Russian ones (қ -> к) first
    C: float = 4.0                     # smaller C = stronger regularization
    class_weight: str | None = None    # None or "balanced"
    min_df: int = 2                    # ignore n-grams seen in fewer than 2 training comments
    sublinear_tf: bool = True          # use 1 + log(count) instead of the raw count
    max_iter: int = 5000
    window: int | None = None          # words per window for long comments, None = whole text
    use_synthetic: bool = True


def prepare(texts, fold: bool) -> list[str]:
    return [normalize(pii.scrub(t), fold=fold) for t in texts]


def make_windows(norm_text: str, window: int | None) -> list[str]:
    """Split a long comment into overlapping windows (each one starts half a window later)."""
    words = norm_text.split()
    if not window or len(words) <= window:
        return [norm_text]
    step = max(1, window // 2)
    starts = range(0, len(words) - window + step, step)
    return [" ".join(words[s:s + window]) for s in starts if words[s:s + window]]


def build_vectorizer(s: Settings) -> TfidfVectorizer:
    return TfidfVectorizer(analyzer=s.analyzer, ngram_range=tuple(s.ngram_range), min_df=s.min_df,
                           sublinear_tf=s.sublinear_tf, lowercase=False, dtype=np.float32)


def build_classifier(s: Settings, seed: int) -> LogisticRegression:
    return LogisticRegression(C=s.C, class_weight=s.class_weight, solver="liblinear",
                              max_iter=s.max_iter, random_state=seed)


def window_scores(vectorizer, clf, norm_texts: list[str], window: int | None) -> np.ndarray:
    """Score every window and keep the highest score per comment.

    So one swear word in a long comment is not diluted by many normal words.
    """
    pieces, owner = [], []
    for i, t in enumerate(norm_texts):
        w = make_windows(t, window)
        pieces.extend(w)
        owner.extend([i] * len(w))
    scores = clf.decision_function(vectorizer.transform(pieces))
    out = np.full(len(norm_texts), -np.inf)
    np.maximum.at(out, np.array(owner), scores)
    return out


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class ProfanityClassifier:
    """Everything needed for prediction. Saved to models/model.joblib."""

    def __init__(self, settings: Settings | None = None, threshold: float = 0.5, seed: int = 42):
        self.settings = settings or Settings()
        self.threshold = float(threshold)
        self.seed = seed
        self.metadata: dict = {}

    def fit(self, texts, y) -> "ProfanityClassifier":
        s = self.settings
        self.vectorizer_ = build_vectorizer(s)
        X = self.vectorizer_.fit_transform(prepare(texts, s.fold))   # vocabulary comes from training text only
        self.clf_ = build_classifier(s, self.seed).fit(X, np.asarray(y))
        return self

    def decision_function(self, texts) -> np.ndarray:
        return window_scores(self.vectorizer_, self.clf_, prepare(texts, self.settings.fold),
                             self.settings.window)

    def predict_proba(self, texts) -> np.ndarray:
        return sigmoid(self.decision_function(texts))

    def predict(self, texts) -> np.ndarray:
        return (self.predict_proba(texts) >= self.threshold).astype(int)

    def explain(self, text: str, top_k: int = 8) -> dict:
        """Every step from raw text to label for one comment (used by predict --explain)."""
        s = self.settings
        scrubbed = pii.scrub(text)
        norm = normalize(scrubbed, fold=s.fold)
        windows = make_windows(norm, s.window)
        X = self.vectorizer_.transform(windows)
        scores = self.clf_.decision_function(X)
        best = int(np.argmax(scores))
        row = X[best].tocoo()
        names = self.vectorizer_.get_feature_names_out()
        coef = self.clf_.coef_[0]
        contrib = sorted(((names[j], float(v), float(coef[j]), float(v * coef[j]))
                          for j, v in zip(row.col, row.data)), key=lambda t: -abs(t[3]))
        known = row.nnz
        analyzer = self.vectorizer_.build_analyzer()
        total = len(set(analyzer(windows[best])))
        prob = float(sigmoid(scores[best]))
        return {
            "raw": text, "after_pii_scrub": scrubbed, "normalized": norm,
            "windows": windows, "scored_window": windows[best],
            "distinct_ngrams_in_window": total, "ngrams_known_from_training": known,
            "top_contributions": [{"ngram": n, "tfidf": round(v, 4), "weight": round(w, 4),
                                   "contribution": round(c, 4)} for n, v, w, c in contrib[:top_k]],
            "intercept": float(self.clf_.intercept_[0]), "score": float(scores[best]),
            "probability": prob, "threshold": self.threshold, "label": int(prob >= self.threshold),
        }

    def describe(self) -> dict:
        return {"settings": asdict(self.settings), "threshold": self.threshold, "seed": self.seed,
                "vocabulary_size": len(self.vectorizer_.vocabulary_), **self.metadata}
