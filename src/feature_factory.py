"""Versioned fold-local TF-IDF and stylometry feature views."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

FEATURE_VERSION = "features-v1"


def _normalize(text: str) -> str:
    """A derived view only; canonical source text is never modified."""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def _shape(text: str) -> str:
    text = _normalize(text)
    text = re.sub(r"\d", "0", text)
    text = re.sub(r"[a-z]", "x", text)
    return re.sub(r"x{5,}", "xxxxx", text)


def stylometry(texts) -> np.ndarray:
    rows = []
    words_re = re.compile(r"\b\w+\b", re.UNICODE)
    for raw in texts:
        text = str(raw)
        words = words_re.findall(text)
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        n_words = max(1, len(words))
        rows.append([
            len(text), len(words), len(sentences), len(text) / n_words,
            len(words) / max(1, len(sentences)), sum(len(w) for w in words) / n_words,
            sum(ch.isupper() for ch in text) / max(1, len(text)),
            sum(ch.isdigit() for ch in text) / max(1, len(text)),
            *(text.count(ch) / max(1, len(text)) for ch in [",", ";", ":", "—", "–", "-", "'", '"', "(", "?", "!"]),
            *(sum(w.lower() == f for w in words) / n_words for f in ["the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "for", "with", "that", "this", "is", "are"]),
        ])
    return np.asarray(rows, dtype=np.float64)


def _config_key(config):
    payload = json.dumps({"version": FEATURE_VERSION, **config}, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


class FoldFeatures:
    """All vocabulary/scaling state is fitted using the supplied training texts."""
    def __init__(self, config=None):
        self.config = {"views": ["char", "word"], **(config or {})}
        self.vectorizers = {}
        self.scaler = None

    def fit_transform(self, train_text):
        texts = [str(t) for t in train_text]
        matrices = []
        for view in self.config["views"]:
            if view == "char":
                vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
                source = texts
            elif view == "word":
                vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, strip_accents="unicode", sublinear_tf=True)
                source = texts
            elif view == "punctuation":
                vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2, sublinear_tf=True, token_pattern=None)
                source = [re.sub(r"[\w\s]", " ", t) for t in texts]
            elif view == "shape":
                vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
                source = [_shape(t) for t in texts]
            elif view == "stylometry":
                self.scaler = StandardScaler()
                matrices.append(csr_matrix(self.scaler.fit_transform(stylometry(texts))))
                continue
            else:
                raise ValueError(f"unknown feature view: {view}")
            matrices.append(vec.fit_transform(source))
            self.vectorizers[view] = vec
        if not matrices:
            raise ValueError("at least one feature view is required")
        return hstack(matrices, format="csr")

    def transform(self, texts):
        texts = [str(t) for t in texts]
        matrices = []
        for view in self.config["views"]:
            if view == "stylometry":
                matrices.append(csr_matrix(self.scaler.transform(stylometry(texts))))
            else:
                source = [_shape(t) for t in texts] if view == "shape" else [re.sub(r"[\w\s]", " ", t) for t in texts] if view == "punctuation" else texts
                matrices.append(self.vectorizers[view].transform(source))
        return hstack(matrices, format="csr")


def build_features(train_text, other_text=None, *, ngram_char=(3, 5), ngram_word=(1, 2), max_features=None, views=None):
    """Compatibility helper for the original char+word SVC interface."""
    cfg = {"views": views or ["char", "word"]}
    features = FoldFeatures(cfg)
    train = features.fit_transform(train_text)
    if other_text is None:
        return train, features
    return train, features.transform(other_text), features


def load_or_fit(train_text, other_text, *, config=None, cache_dir=None, fold_token=""):
    """Cache only fold-fitted objects/features; cache key includes fit rows and recipe."""
    config = {"views": ["char", "word"], **(config or {})}
    train_text = [str(t) for t in train_text]
    other_text = [str(t) for t in other_text]
    payload = "\0".join(train_text).encode()
    validation_payload = "\0".join(other_text).encode()
    key = hashlib.sha256(payload + b"\1" + validation_payload + _config_key(config).encode() + str(fold_token).encode()).hexdigest()
    path = Path(cache_dir) / f"{key}.joblib" if cache_dir else None
    if path and path.exists():
        import joblib
        obj = joblib.load(path)
        return obj["train"], obj["other"], obj["features"], key
    features = FoldFeatures(config)
    train = features.fit_transform(train_text)
    other = features.transform(other_text)
    if path:
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"train": train, "other": other, "features": features}, path)
    return train, other, features, key
