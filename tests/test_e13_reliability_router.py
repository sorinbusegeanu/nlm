import numpy as np

from src import e13_reliability_router as e13


def _toy_probabilities(groups=8, checkpoints=4):
    rng = np.random.default_rng(42)
    truth = np.tile(np.arange(len(e13.LABELS)), groups)
    n = len(truth)
    probabilities = rng.random((checkpoints, n, len(e13.LABELS))) * 0.05
    for model in range(checkpoints):
        predicted = truth.copy()
        wrong = (np.arange(n) + model) % 5 == 0
        predicted[wrong] = (predicted[wrong] + model + 1) % len(e13.LABELS)
        probabilities[model, np.arange(n), predicted] += 1.0
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    return probabilities, truth


def test_reliability_statistics_use_canonical_confusion_orientation_and_smoothing():
    truth = np.arange(len(e13.LABELS))
    hard = truth[None, :].copy()
    hard[0, 0] = 1
    stats = e13.reliability_statistics(hard, truth)
    assert stats["counts"][0, 0, 1] == 1
    assert stats["precision"][0, 1] == 2 / 4
    assert stats["recall"][0, 0] == 1 / 3
    np.testing.assert_allclose(stats["directional"].sum(axis=2), 1.0)


def test_roster_starts_at_highest_balanced_accuracy_then_adds_coverage():
    truth = np.arange(len(e13.LABELS))
    hard = np.stack([truth.copy(), truth.copy(), truth.copy()])
    hard[0, :3] = [1, 0, 3]
    hard[1, 2:4] = [3, 2]
    hard[2] = hard[0]
    roster, steps = e13.select_roster(hard, truth, ["a", "b", "a"],
                                      ["N1", "N2", "N3"], 3)
    assert roster[0] == 1
    assert len(roster) == 3
    assert steps[1]["newly_covered"] > 0


def test_roster_exclusion_removes_members_without_replacement():
    truth = np.arange(len(e13.LABELS))
    hard = np.stack([truth.copy(), truth.copy(), truth.copy()])
    roster, steps = e13.select_roster(hard, truth, ["a", "b", "c"],
                                      ["N1", "N2", "N3"], 3, ["N1"])
    assert roster == [1, 2]
    assert [step["checkpoint_id"] for step in steps] == ["N2", "N3"]


def test_feature_rows_align_row_major_with_checkpoint_correctness_targets():
    probabilities, truth = _toy_probabilities(groups=2)
    fit = np.arange(18)
    held = np.arange(18, 36)
    frame, top = e13.build_feature_frame(
        probabilities, ["N1", "N2", "N3", "N4"], ["a", "a", "b", "c"],
        truth[fit], fit, held)
    target = e13._targets(probabilities[:, held].argmax(axis=2), truth[held])
    assert len(frame) == len(target) == len(held) * probabilities.shape[0]
    assert frame.checkpoint_id.iloc[:4].tolist() == ["N1", "N2", "N3", "N4"]
    assert frame.predicted_label.tolist() == e13.CLASSES[top].T.reshape(-1).tolist()
    assert set(e13.CATEGORICAL).issubset(frame.columns)
    assert np.isfinite(frame.select_dtypes(include=["number"]).to_numpy()).all()


def test_family_max_aggregation_and_deterministic_ties():
    # Two members of family A cannot outvote a single family B by duplication.
    hard = np.asarray([[0], [0], [1]])
    scores = np.asarray([[0.8], [0.7], [0.75]])
    ensemble = np.full((1, len(e13.LABELS)), 1 / len(e13.LABELS))
    pred, selected = e13.route_predictions("family_max", hard, scores, ensemble,
                                           ["a", "a", "b"])
    assert pred.tolist() == [0]
    assert selected.tolist() == [0]


def test_outer_held_labels_do_not_change_router_predictions(monkeypatch):
    probabilities, truth = _toy_probabilities()
    groups = np.repeat([f"g{i}" for i in range(8)], len(e13.LABELS))
    row_ids = np.asarray([f"row-{i}" for i in range(len(truth))])
    ids = [f"N{i:04d}" for i in range(probabilities.shape[0])]
    families = ["a", "a", "b", "c"]
    fit, held = np.arange(6 * len(e13.LABELS)), np.arange(6 * len(e13.LABELS), len(truth))
    params = dict(e13.CATBOOST_PARAMS)
    params.update(iterations=5, depth=3, thread_count=1, od_wait=2)
    monkeypatch.setattr(e13, "CATBOOST_PARAMS", params)
    first = e13._fit_router(probabilities, ids, families, truth.copy(), groups,
                            fit, held, 3, "checkpoint", np.zeros(len(truth), dtype=int), row_ids)
    changed = truth.copy()
    changed[held] = (changed[held] + 1) % len(e13.LABELS)
    second = e13._fit_router(probabilities, ids, families, changed, groups,
                             fit, held, 3, "checkpoint", np.zeros(len(truth), dtype=int), row_ids)
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1]["checkpoint_ids"] == second[1]["checkpoint_ids"]
    for inner in first[1]["inner_fit_folds"]:
        assert set(inner["fit_ids"]).isdisjoint(inner["held_ids"])
