import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from src import confusion_ensemble as evidence


def test_confusion_orientation_smoothing_and_uniform_prior():
    hard = np.array([[0, 1, 1, 0, 1]])
    truth = np.array([0, 0, 0, 1, 1])
    counts, correlations = evidence._fit_statistics(hard, truth, 2)
    np.testing.assert_array_equal(counts[0], [[1, 2], [1, 1]])
    likelihoods, weights = evidence._parameters(counts, correlations, alpha=0.5, gamma=1)
    np.testing.assert_allclose(likelihoods[0], [[0.375, 0.625], [0.5, 0.5]])
    np.testing.assert_allclose(likelihoods.sum(axis=2), 1)
    # A predicted class zero is more likely under true class one for this checkpoint.
    actual = evidence._predict_codes(np.array([[0, 1]]), likelihoods, weights, [0.5, 0.5])
    np.testing.assert_array_equal(actual, [1, 0])


def test_positive_correlations_discount_duplicates_and_handle_constants():
    truth = np.zeros(4, dtype=int)
    hard = np.array([[0, 1, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 0, 0]])
    correlations = evidence.error_correlations(hard, truth)
    np.testing.assert_allclose(correlations, [[0, 1, 0, 0], [1, 0, 0, 0],
                                            [0, 0, 0, 0], [0, 0, 0, 0]])
    counts = np.ones((4, 2, 2))
    _, weights = evidence._parameters(counts, correlations, alpha=1, gamma=2)
    np.testing.assert_allclose(weights, [1 / 3, 1 / 3, 1, 1])
    _, unpenalized = evidence._parameters(counts, correlations, alpha=1, gamma=0)
    np.testing.assert_array_equal(unpenalized, np.ones(4))


def test_coverage_selector_adds_complement_and_stops_at_limit_or_saturation():
    truth = np.tile([0, 1], 10)
    first, second = truth.copy(), truth.copy()
    first[:2] = 1 - first[:2]
    second[2:6] = 1 - second[2:6]
    hard = np.stack([first, second, second])
    roster, trace = evidence.select_coverage_roster(hard, truth, 2, limit=15)
    assert roster == [0, 1]
    assert [step["newly_covered"] for step in trace] == [18, 2]
    limited, _ = evidence.select_coverage_roster(hard, truth, 2, limit=1)
    assert limited == [0]
    with pytest.raises(ValueError, match="every canonical label"):
        evidence.select_coverage_roster(hard[:, :1], truth[:1], 2, limit=1)


def test_saved_prediction_respects_checkpoint_and_label_order():
    model = {
        "labels": ["second", "first"], "checkpoint_ids": ["N0002", "N0001"],
        "prior": [0.5, 0.5], "weights": [1, 0.25],
        "confusion_matrices": [[[0.9, 0.1], [0.1, 0.9]], [[0.6, 0.4], [0.4, 0.6]]],
    }
    probabilities = {"N0001": np.eye(2)[[1, 0]], "N0002": np.eye(2)}
    predicted = evidence.predict_confusion_ensemble(json.loads(json.dumps(model)), probabilities,
                                                    labels=["first", "second"])
    np.testing.assert_array_equal(predicted, ["first", "second"])
    with pytest.raises(ValueError, match="canonical labels"):
        evidence.predict_confusion_ensemble(model, probabilities, labels=["a", "b"])


def test_tuning_ties_prefer_stronger_smoothing_and_penalty():
    truth = np.tile([0, 1], 6)
    groups = np.repeat(np.arange(6), 2)
    strategy = {"max_checkpoints": 15, "inner_folds": 3,
                "alpha_grid": [0.5, 1, 2, 5], "gamma_grid": [0, 0.5, 1, 2]}
    model = evidence.fit_confusion_ensemble(np.stack([truth, truth]), truth, groups,
                                           np.arange(12), ["N1", "N2"], strategy,
                                           labels=["a", "b"])
    assert (model["alpha"], model["gamma"]) == (5, 2)
    assert model["checkpoint_ids"] == ["N1"]
    assert all(score["accuracy"] == 1 for score in model["inner_scores"])


def test_outer_held_labels_do_not_change_fitted_parameters_and_inner_groups_are_disjoint():
    rng = np.random.default_rng(42)
    labels = list(evidence.LABELS)
    truth = np.tile(np.arange(len(labels)), 9)
    frame = pd.DataFrame({"id": [f"row-{i}" for i in range(len(truth))],
                          "prompt_id": np.repeat(np.arange(9), len(labels)),
                          "model": np.asarray(labels)[truth]})
    probabilities = {}
    for sid in ["N10", "N2", "N1"]:
        hard = truth.copy()
        changed = rng.random(len(truth)) < 0.35
        hard[changed] = rng.integers(len(labels), size=changed.sum())
        probabilities[sid] = np.eye(len(labels))[hard]
    strategy = {"max_checkpoints": 15, "inner_folds": 3,
                "alpha_grid": [0.5, 2], "gamma_grid": [0, 1]}
    first, details = evidence.evaluate_confusion_ensemble(strategy, 3, frame, probabilities)
    _, held = next(GroupKFold(n_splits=3).split(frame, truth, frame.prompt_id))
    perturbed = frame.copy()
    perturbed.loc[held, "model"] = np.asarray(labels)[(truth[held] + 1) % len(labels)]
    again, changed = evidence.evaluate_confusion_ensemble(strategy, 3, perturbed, probabilities)
    # Oracle diagnostics use held labels, while all fitting fields must be identical.
    fit_fields = set(details["folds"][0]) - {"oracle_correct", "oracle_accuracy"}
    assert {key: details["folds"][0][key] for key in fit_fields} == {
        key: changed["folds"][0][key] for key in fit_fields}
    np.testing.assert_array_equal(first[held], again[held])
    prompt_by_id = frame.set_index("id").prompt_id.to_dict()
    for model in [details["deployment"], *details["folds"]]:
        assert len(model["checkpoint_ids"]) <= 15
        fitting_ids = set(model["fit_ids"])
        assert fitting_ids.isdisjoint(model.get("held_ids", []))
        for inner in model["inner_folds"]:
            assert set(inner["fit_ids"]) | set(inner["held_ids"]) == fitting_ids
            assert {prompt_by_id[i] for i in inner["fit_ids"]}.isdisjoint(
                {prompt_by_id[i] for i in inner["held_ids"]})
    restored = json.loads(json.dumps(details["deployment"], allow_nan=False))
    deployed = evidence.predict_confusion_ensemble(restored, probabilities)
    assert np.mean(deployed == frame.model.to_numpy()) == restored["fit_accuracy"]
