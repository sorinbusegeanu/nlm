import numpy as np
from sklearn.linear_model import LogisticRegression

from src import e13_reliability_router as e13
from src import e20
from src.e28 import family_max_label_scores, finite_assignment_scores, route_family_max
from src.e28_deploy import (
    checkpoint_bundle_relative_path,
    distribute_checkpoints,
    ensure_binary_logistic_predict_proba_compat,
    intervals_overlap,
)
from src.v2_checkpoint_trust import LABELS


def test_binary_logistic_compat_restores_legacy_attribute_without_changing_scores():
    rng = np.random.RandomState(17)
    features = rng.normal(size=(80, 5)).astype(np.float32)
    labels = (features[:, 0] + features[:, 1] > 0).astype(np.int64)
    model = LogisticRegression(random_state=17).fit(features, labels)
    expected = model.predict_proba(features)
    if hasattr(model, "multi_class"):
        del model.multi_class

    assert ensure_binary_logistic_predict_proba_compat(model) is model
    assert model.multi_class == "auto"
    np.testing.assert_array_equal(model.predict_proba(features), expected)


def test_checkpoint_paths_are_resolved_beneath_the_bundle_root():
    assert checkpoint_bundle_relative_path("checkpoints/N0082/model.joblib") == "checkpoints/N0082/model.joblib"
    assert checkpoint_bundle_relative_path("N0030") == "checkpoints/N0030"
    assert checkpoint_bundle_relative_path("N0084/svc_foldvalidation.joblib") == "checkpoints/N0084/svc_foldvalidation.joblib"
    import pytest
    with pytest.raises(ValueError, match="bundle-relative"):
        checkpoint_bundle_relative_path("../outside/model.cbm")


def test_family_max_uses_max_within_family_then_mean_across_families():
    hard = np.asarray([[0], [0], [1]])
    reliability = np.asarray([[0.4], [0.9], [0.6]])
    scores = family_max_label_scores(hard, reliability, ["a", "a", "b"])
    assert scores.shape == (1, 18)
    assert scores[0, 0] == 0.9
    assert scores[0, 1] == 0.6
    assert np.isneginf(scores[0, 2:]).all()


def test_route_ties_and_canonical_label_order_match_original():
    hard = np.asarray([[0, 2], [1, 2], [0, 3]])
    reliability = np.asarray([[0.7, 0.4], [0.7, 0.4], [0.7, 0.4]])
    probs = np.zeros((2, len(LABELS)), dtype=float)
    probs[:, 1] = 0.8
    probs[:, 0] = 0.2
    probs[1, 3] = 0.9
    pred, _, scores, finite = route_family_max(hard, reliability, probs,
                                               ["family-a", "family-b", "family-a"])
    old_pred, _ = e13.route_predictions("family_max", hard, reliability, probs,
                                        ["family-a", "family-b", "family-a"])
    np.testing.assert_array_equal(pred, old_pred)
    assert pred.tolist() == [1, 3]
    assert scores.shape == finite.shape == (2, len(LABELS))
    assert np.isfinite(finite).all()


def test_unsupported_labels_get_finite_lower_scores_and_probability_fallback():
    raw = np.full((1, len(LABELS)), -np.inf)
    raw[0, 4] = 0.2
    raw[0, 6] = 0.7
    probs = np.zeros_like(raw)
    probs[0, 2] = 0.4
    probs[0, 3] = 0.8
    safe = finite_assignment_scores(raw, probs)
    assert np.isfinite(safe).all()
    assert np.isclose(safe[0, 4], 0.2) and np.isclose(safe[0, 6], 0.7)
    assert safe[0, 3] > safe[0, 2]
    assert safe[0, 3] < raw[0, 4]


def test_hungarian_singletons_and_class_order_are_deterministic():
    scores = np.zeros((2, len(LABELS)), dtype=float)
    scores[0, 2] = 0.8
    groups = np.asarray(["single-a", "single-b"])
    result = e20.hungarian(scores, groups, ["row-a", "row-b"])
    assert result.tolist() == [2, 0]
    # Equal scores within a multirow group resolve in canonical column order.
    result = e20.hungarian(np.zeros((2, len(LABELS))), ["same", "same"], ["a", "b"])
    assert result.tolist() == [0, 1]


def test_two_gpu_schedule_uses_estimates_and_requires_both_workers():
    checkpoints = [
        {"candidate_id": "N1", "model": "large", "files": [{"size": 1000}]},
        {"candidate_id": "N2", "model": "small", "files": [{"size": 100}]},
        {"candidate_id": "N3", "model": "small", "files": [{"size": 100}]},
    ]
    plan = distribute_checkpoints(checkpoints, {"N1": 20, "N2": 9, "N3": 8},
                                  {"large": 1, "small": 1}, 64)
    assert len(plan) == 2 and all(part["checkpoint_ids"] for part in plan)
    assert {sid for part in plan for sid in part["checkpoint_ids"]} == {"N1", "N2", "N3"}
    assert intervals_overlap([
        {"gpu_id": 0, "start_epoch": 10, "end_epoch": 30},
        {"gpu_id": 1, "start_epoch": 20, "end_epoch": 40},
    ])
    assert not intervals_overlap([{"gpu_id": 0, "start_epoch": 10, "end_epoch": 30}])
