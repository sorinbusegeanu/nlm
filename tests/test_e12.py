import numpy as np
import pytest
from scipy.special import expit

from src import e12
from src import v2_checkpoint_trust


def test_pair_features_and_classifier_are_symmetric():
    rng = np.random.default_rng(9)
    probabilities = rng.random((3, 36, 18))
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    truth = np.tile(np.arange(18), 2)
    stats = e12.reliability(probabilities.argmax(axis=2), truth)
    families = ["a", "b", "b"]
    a, b = np.arange(36) % 18, (np.arange(36) + 2) % 18
    forward = e12.pair_features(probabilities, stats, families, a, b)
    backward = e12.pair_features(probabilities, stats, families, b, a)
    np.testing.assert_allclose(forward, -backward)
    scaler, model = e12.fit_classifier(forward, np.arange(36) % 2, C=0.1)
    p = model.predict_proba(scaler.transform(forward))[:, 1]
    q = model.predict_proba(scaler.transform(backward))[:, 1]
    np.testing.assert_allclose(p + q, 1, atol=1e-12)
    np.testing.assert_allclose(p, expit(scaler.transform(forward) @ model.coef_[0]))


def test_directional_confusion_and_smoothed_precision_recall():
    truth = np.arange(18)
    hard = truth.copy()
    hard[0] = 1
    stats = e12.reliability(hard[None, :], truth)
    assert stats["counts"][0, 0, 1] == 1
    assert stats["counts"][0, 1, 0] == 0
    assert stats["directional"][0, 0, 1] == 2 / 19
    assert stats["directional"][0, 1, 0] == 1 / 19
    assert stats["precision"][0, 1] == 2 / 4
    assert stats["recall"][0, 0] == 1 / 3
    np.testing.assert_allclose(stats["directional"].sum(axis=2), 1)


def test_roster_prefers_complementary_errors_then_new_families():
    truth = np.arange(18)
    first = truth.copy()
    first[[0, 1]] = [1, 0]
    duplicate = first.copy()
    different_family = first.copy()
    hard = np.stack([first, duplicate, different_family])
    roster, steps = e12.select_roster(hard, truth, ["a", "a", "b"], 3)
    assert roster == [0, 2, 1]
    assert steps[1]["new_family"]
    assert len(roster) == 3
    with pytest.raises(ValueError, match="canonical labels"):
        e12.select_roster(hard[:, :5], truth[:5], ["a", "a", "b"], 3)


def test_ties_keep_original_candidate_order():
    truth = np.arange(18)
    probs = np.ones((2, len(truth), 18)) / 18
    stats = e12.reliability(probs.argmax(axis=2), truth)

    class EqualClassifier:
        def predict_proba(self, X):
            return np.full((len(X), 2), 0.5)

    class IdentityScaler:
        def transform(self, X):
            return X

    pred, candidates, _ = e12.arbitrate(probs, stats, ["a", "b"], 3, IdentityScaler(), EqualClassifier())
    np.testing.assert_array_equal(candidates[0], [0, 1, 2])
    np.testing.assert_array_equal(pred, np.zeros(len(truth), dtype=int))


def test_nested_fit_isolated_from_outer_held_labels():
    rng = np.random.default_rng(29)
    truth = np.tile(np.arange(18), 9)
    groups = np.repeat(np.arange(9), 18)
    ids = np.asarray([f"id-{i}" for i in range(len(truth))])
    probs = rng.random((4, len(truth), 18))
    for model in range(4):
        probs[model, np.arange(len(truth)), truth] += 1.2
    probs /= probs.sum(axis=2, keepdims=True)
    hard = probs.argmax(axis=2)
    fit = np.flatnonzero(groups < 6)
    held = np.flatnonzero(groups >= 6)
    args = (probs, hard, truth, groups, ids, ["N1", "N2", "N3", "N4"], ["a", "b", "c", "d"], fit, held, 3, 3)
    first = e12.nested_fit(*args)
    mutated = truth.copy()
    mutated[held] = (mutated[held] + 1) % 18
    second = e12.nested_fit(probs, hard, mutated, *args[3:])
    np.testing.assert_array_equal(first[0], second[0])
    assert first[3] == second[3]
    for fold in first[3]["inner_folds"]:
        assert set(fold["fit_ids"]).isdisjoint(fold["held_ids"])
        assert set(fold["fit_ids"]) | set(fold["held_ids"]) == set(ids[fit])
        assert len(fold["checkpoint_ids"]) == 3


def test_probability_loader_rejects_misaligned_rows(tmp_path, monkeypatch):
    import pandas as pd

    path = tmp_path / v2_checkpoint_trust.PREDICTION_DIR / "example.parquet"
    path.parent.mkdir(parents=True)
    frame = pd.DataFrame({"id": ["row-b", "row-a"], "y_true": ["x", "y"]})
    for label in e12.LABELS:
        frame[f"prob::{label}"] = 1 / len(e12.LABELS)
    frame.to_parquet(path, index=False)
    monkeypatch.setattr(v2_checkpoint_trust, "_find_manifest", lambda *args: (_ for _ in ()).throw(FileNotFoundError()))
    with pytest.raises(ValueError, match="not aligned"):
        v2_checkpoint_trust._read_saved_probabilities(tmp_path, "example", ["row-a", "row-b"], ["y", "x"])


def test_corrected_introduced_and_candidate_coverage_are_aligned():
    truth = np.array([0, 1, 2, 3])
    baseline = np.array([1, 1, 0, 3])
    prediction = np.array([0, 2, 2, 3])
    candidates = np.array([[0, 1, 2], [1, 2, 0], [0, 2, 1], [3, 0, 1]])
    result = e12.metrics(truth, prediction, candidates, 2, baseline)
    assert result["accuracy"] == 0.75
    assert result["candidate_coverage"] == 1
    assert result["corrected"] == 2
    assert result["introduced"] == 1


def test_catalog_entry_runs_fold_local_evaluation_and_fits_deployment():
    import pandas as pd
    from src.ensemble_catalog import _evaluate_strategy

    rng = np.random.default_rng(18)
    truth = np.tile(np.arange(18), 9)
    frame = pd.DataFrame({"id": [f"row-{i}" for i in range(len(truth))],
                          "prompt_id": np.repeat(np.arange(9), 18).astype(str),
                          "model": np.asarray(e12.LABELS)[truth]})
    probabilities, members = {}, {}
    for number in range(1, 5):
        sid = f"N{number:04d}"
        p = rng.random((len(truth), 18))
        p[np.arange(len(truth)), truth] += 1
        probabilities[sid] = p / p.sum(axis=1, keepdims=True)
        members[sid] = {"model": f"family_{number}"}
    strategy = {"id": "E12-1", "algorithm": "pairwise_arbitration", "candidates": 2,
                "max_checkpoints": 3, "inner_folds": 3, "C_grid": [0.1]}
    pred, result = _evaluate_strategy(strategy, {"folds": 3}, frame, members, probabilities)
    assert len(pred) == len(frame)
    assert len(result["folds"]) == 3
    assert len(result["deployment"]["checkpoint_ids"]) == 3
    assert all(len(fold["checkpoint_ids"]) == 3 for fold in result["folds"])
    assert all([row["C"] for row in fold["tuning"]] == [0.1] for fold in result["folds"])
    assert set(result["per_label"]) == set(e12.LABELS)
    assert result["candidate_coverage"] >= result["accuracy"]
