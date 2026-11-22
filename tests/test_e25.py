import numpy as np

from src import e22
from src.e22 import build_features
from src.e25 import _aligned_features, _fast_features, _global_deployment_rosters
from src.v2_checkpoint_trust import LABELS


def _ctx():
    rng = np.random.default_rng(25)
    n, cp, k = 36, 6, len(LABELS)
    y = np.arange(n) % k
    probabilities = rng.random((cp, n, k))
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    for row, label in enumerate(y):
        probabilities[row % cp, row, label] += 2.0
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    return {"matrices": probabilities.astype(np.float32),
        "checkpoint_ids": [f"N{i:04d}" for i in range(cp)],
        "families": [f"family-{i % 3}" for i in range(cp)], "y": y,
        "row_ids": np.asarray([f"r{i}" for i in range(n)])}


def test_vectorized_e25_features_match_e22_and_preserve_row_order():
    ctx = _ctx()
    rows = np.asarray([7, 2, 11])
    pool = list(range(6))
    e22._BASE_FEATURE_CACHE.clear()
    expected = build_features(ctx, rows, pool)
    actual = _fast_features(ctx, rows, pool)
    assert set(expected.columns) == set(actual.columns)
    assert expected.candidate_label.tolist() == actual.candidate_label.tolist()
    for column in expected.columns:
        if column != "candidate_label":
            assert np.allclose(expected[column].to_numpy(float), actual[column].to_numpy(float))


def test_dropped_checkpoint_recomputes_aggregates_and_zero_fills_absent_channels():
    ctx = _ctx()
    rows = np.asarray([1, 3, 5])
    full = _fast_features(ctx, rows, list(range(6)))
    schema = list(full.columns)
    reduced = _aligned_features(ctx, rows, [0, 2, 3, 4, 5], schema)
    assert np.all(reduced["cp::N0001::prob"].to_numpy() == 0)
    assert np.all(reduced["cp::N0001::present"].to_numpy() == 0)
    assert np.all(reduced["candidate_checkpoint_votes"].to_numpy() <=
                  full["candidate_checkpoint_votes"].to_numpy())
    assert np.all(reduced["family_count"].to_numpy() == 3)


def test_deployment_rosters_obey_global_distinct_checkpoint_budgets():
    folds = {}
    ids = [f"N{i:04d}" for i in range(80)]
    for fold in range(5):
        folds[str(fold)] = {"target_rosters": {
            str(budget): ids[fold:fold + budget] for budget in (60, 40, 30, 25, 20)}}
    # Test with a candidate universe larger than each requested limit.
    rosters = _global_deployment_rosters(folds)
    for budget in (60, 40, 30, 25, 20):
        assert rosters[str(budget)]["count"] <= budget
        assert len(rosters[str(budget)]["checkpoint_ids"]) == len(set(rosters[str(budget)]["checkpoint_ids"]))

