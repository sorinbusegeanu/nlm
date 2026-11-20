import numpy as np

from src.e16 import _pair_features, decode_graph, _training_pairs
from src.v2_checkpoint_trust import LABELS
from sklearn.feature_extraction.text import TfidfVectorizer


def test_pair_features_include_text_similarity_and_structure():
    texts = ["A short list:\n- one\n- two", "A short list:\n- one\n- two", "Different prose with words"]
    char = TfidfVectorizer(analyzer="char", ngram_range=(2, 4)).fit_transform(texts)
    word = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(texts)
    x = _pair_features(texts, char, word, np.array([0, 0]), np.array([1, 2]))
    assert x.shape == (2, 17)
    assert x[0, 0] > x[1, 0]
    assert np.isfinite(x).all()


def test_pair_sampler_uses_same_prompt_positives_and_different_prompt_negatives():
    texts = ["shared response style alpha", "shared response style beta",
             "shared response style alpha", "another content style"]
    groups = np.array(["p1", "p1", "p2", "p3"])
    char = TfidfVectorizer(analyzer="char", ngram_range=(2, 4)).fit_transform(texts)
    positive, negative = _training_pairs(texts, groups, char, neighbors=2)
    assert len(positive) > 0 and len(negative) > 0
    assert all(groups[a] == groups[b] for a, b in positive)
    assert all(groups[a] != groups[b] for a, b in negative)


def test_text_only_graph_decoder_applies_soft_uniqueness():
    n = 2
    scores = np.full((n, len(LABELS)), -1.0)
    scores[:, 0] = [0.9, 0.8]
    scores[:, 1] = [0.75, 0.72]
    edges = (np.array([0]), np.array([1]), np.array([0.99]))
    pred, graph = decode_graph(np.array(["row-b", "row-a"]), scores, edges,
                               threshold=0.8, penalty=0.5)
    assert pred[0] != pred[1]
    assert graph["active_edges"] == 1
    assert graph["active_nodes"].all()
