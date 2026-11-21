import json
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src import e19f


def _toy_vectors():
    return np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]], dtype=np.float32)


def test_complete_link_and_mnn_grouping_are_deterministic_under_input_order():
    ids = np.asarray(["r-a", "r-b", "r-c", "r-d"])
    embeddings = _toy_vectors()
    cosine = embeddings @ embeddings.T
    expected_a = e19f.cluster_cosine_complete(ids, cosine, threshold=0.9)[0]
    expected_b = e19f.cluster_mutual_complete(ids, cosine, threshold=0.9, k=1)[0]

    order = np.asarray([2, 0, 3, 1])
    actual_a = e19f.cluster_cosine_complete(ids[order], cosine[np.ix_(order, order)], threshold=0.9)[0]
    actual_b = e19f.cluster_mutual_complete(ids[order], cosine[np.ix_(order, order)], threshold=0.9, k=1)[0]

    assert dict(zip(ids, expected_a)) == dict(zip(ids[order], actual_a))
    assert dict(zip(ids, expected_b)) == dict(zip(ids[order], actual_b))
    assert len(set(expected_a)) == 2
    assert len(set(expected_b)) == 2


def test_mnn_complete_link_blocks_weak_transitive_bridge():
    angles = np.deg2rad([0.0, 30.0, 60.0])
    embeddings = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
    cosine = embeddings @ embeddings.T
    groups, _ = e19f.cluster_mutual_complete(["a", "b", "c"], cosine,
        threshold=0.75, k=2)

    assert groups[0] == groups[1]
    assert groups[2] != groups[0]


def test_word_character_embedding_candidates_and_pair_scorer_are_finite():
    ids = np.asarray(["a", "b", "c", "d"])
    texts = np.asarray(["write a short limerick about a comet",
        "write a brief limerick about a comet", "solve this geometry problem",
        "find the area of the triangle"])
    embeddings = np.asarray([[1, .1], [.99, .1], [.1, 1], [.1, .99]], dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    vectorizers = e19f._fit_lexical(texts, np.arange(4))
    lexical = e19f._transform_lexical(texts, vectorizers)
    rows, pairs = e19f.candidate_pairs(np.arange(4), ids, lexical, embeddings, neighbors=3)

    assert len(pairs) > 0
    features = e19f._pair_features(rows, pairs, lexical, embeddings)
    assert features.shape == (len(pairs), 3)
    assert np.isfinite(features).all()


def test_embedding_cache_resumes_by_response_hash_without_reloading_model(monkeypatch, tmp_path):
    import torch
    import transformers

    class Batch(dict):
        def to(self, _device):
            return self

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __call__(self, texts, **kwargs):
            values = [sum(str(text).encode("utf-8")) % 7 + 1 for text in texts]
            ids = torch.tensor([[v, v + 1] for v in values], dtype=torch.long)
            return Batch(input_ids=ids, attention_mask=torch.ones_like(ids))

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(self, input_ids, **kwargs):
            x = input_ids.float()
            hidden = torch.stack((x, torch.ones_like(x)), dim=-1)
            return SimpleNamespace(last_hidden_state=hidden)

    monkeypatch.setattr(transformers, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(transformers, "AutoModel", FakeModel)
    cache = tmp_path / "vectors.sqlite3"

    first, metadata = e19f.encode_response_embeddings(["alpha", "beta", "alpha"],
        cache_path=cache, device="cpu", batch_size=2, progress=False)
    assert metadata["newly_encoded_hashes"] == 2
    assert first.shape == (3, 2)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)
    assert np.array_equal(first[0], first[2])

    class MustNotLoad:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise AssertionError("cached vectors should resume without loading the model")

    monkeypatch.setattr(transformers, "AutoTokenizer", MustNotLoad)
    monkeypatch.setattr(transformers, "AutoModel", MustNotLoad)
    resumed, resumed_meta = e19f.encode_response_embeddings(["beta", "alpha"],
        cache_path=cache, device="cpu", progress=False)
    assert resumed_meta["newly_encoded_hashes"] == 0
    assert np.array_equal(resumed, first[[1, 0]])


def test_kaggle_predict_api_uses_only_ids_text_and_embeddings(tmp_path):
    model = {"method": "E19-F-A", "selected": {"threshold": 0.9}}
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    embeddings = _toy_vectors()

    groups = e19f.predict_groups(["a", "b", "c", "d"],
        ["response a", "response b", "response c", "response d"],
        embeddings=embeddings, model_path=path)

    assert len(groups) == 4
    assert groups[0] == groups[1]
    assert groups[2] == groups[3]
    with pytest.raises(ValueError, match="unique aligned"):
        e19f.predict_groups(["a", "a"], ["x", "y"], embeddings=embeddings[:2], model_path=path)
