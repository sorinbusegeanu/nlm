import numpy as np

from src import e20d_kaggle_failure_analysis as analysis


def test_fold_score_aggregation_is_explicit_and_row_aligned(monkeypatch):
    monkeypatch.setattr(analysis.e20, "hungarian", lambda scores, groups: scores.argmax(axis=1))
    # Three rows, two labels, two fold models; first dimension is model.
    fold_a = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    fold_b = np.array([[0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    groups = np.array(["a", "b", "c"])
    raw, independent, hungarian = analysis._aggregate([fold_a, fold_b], groups,
                                                       np.array([0, 1, 0]))
    normalized, _, _ = analysis._aggregate([fold_a, fold_b], groups,
                                            np.array([0, 1, 0]), normalize=True)
    np.testing.assert_allclose(raw, (fold_a + fold_b) / 2)
    np.testing.assert_array_equal(independent, raw.argmax(axis=1))
    np.testing.assert_array_equal(hungarian, independent)
    np.testing.assert_allclose(normalized.sum(axis=1), 1.0)


def test_reconstructed_group_builder_only_receives_selected_rows(monkeypatch):
    captured = {}
    monkeypatch.setattr(analysis.e19b_group, "transform_similarity_spaces",
                        lambda reconstructed, texts, vectorizers: {"dummy": 1})

    def candidates(rows, spaces, embeddings, **kwargs):
        captured["rows"] = np.asarray(rows).copy()
        # pair indices are local to rows, matching e19b_group's contract.
        return np.array([[0, 1]], dtype=np.int64)

    monkeypatch.setattr(analysis.e19b_group, "candidate_pairs", candidates)
    monkeypatch.setattr(analysis.e19b_group, "pair_features",
                        lambda *args: np.zeros((1, len(analysis.e19b_group.FEATURE_NAMES))))
    monkeypatch.setattr(analysis.e19b_group, "score_pairs", lambda *args: np.array([0.9]))

    def cluster(ids, pairs, probabilities, **kwargs):
        assert list(ids) == ["r1", "r4"]
        return np.array(["g0", "g0"]), {"clusters": []}

    monkeypatch.setattr(analysis.e19b_group, "cluster_complete_link", cluster)
    ctx = {"row_ids": np.array(["r0", "r1", "r2", "r3", "r4"]),
           "texts": np.array(["a", "b", "c", "d", "e"])}
    data = {"recon": np.array(["a", "b", "c", "d", "e"]),
            "embeddings": np.ones((5, 3), dtype=np.float32),
            "topics": np.array([""] * 5), "details": [[] for _ in range(5)],
            "constraints": [[] for _ in range(5)]}
    groups, pairs, _ = analysis._load_package_groups(ctx, data,
        {"vectorizers": {}, "pair_scorer": {}, "threshold": 0.5},
        np.array([1, 4]))
    np.testing.assert_array_equal(captured["rows"], [1, 4])
    np.testing.assert_array_equal(groups, ["g0", "g0"])
    assert pairs[("r1", "r4")] == 0.9


def test_normalized_average_normalizes_each_model_before_mean(monkeypatch):
    monkeypatch.setattr(analysis.e20, "hungarian", lambda scores, groups: scores.argmax(axis=1))
    a = np.array([[9.0, 1.0]])
    b = np.array([[1.0, 9.0]])
    groups = np.array(["singleton"])
    raw, _, _ = analysis._aggregate([a, b], groups, np.array([0]))
    norm, _, _ = analysis._aggregate([a, b], groups, np.array([0]), normalize=True)
    np.testing.assert_allclose(raw, [[5.0, 5.0]])
    np.testing.assert_allclose(norm, [[0.5, 0.5]])
