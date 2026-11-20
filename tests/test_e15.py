import numpy as np
from sklearn.model_selection import GroupKFold

from src.e15 import (build_features, select_candidates, _apply_threshold,
                     _fit_logistic, _probability)
from src.v2_checkpoint_trust import LABELS


def _synthetic_inputs():
    rng = np.random.default_rng(11)
    p = rng.random((3, 6, len(LABELS)))
    p /= p.sum(axis=2, keepdims=True)
    roster = np.asarray([0, 1])
    hard = p[roster].argmax(axis=2)
    reliability = np.asarray([[.8, .7, .6, .9, .5, .8],
                              [.7, .8, .9, .6, .8, .7]])
    ensemble = p[roster].mean(axis=0)
    ensemble /= ensemble.sum(axis=1, keepdims=True)
    return p, roster, hard, reliability, ensemble


def test_candidate_selection_is_deterministic_and_distinct():
    _, _, hard, rel, ensemble = _synthetic_inputs()
    a = select_candidates(hard, rel, ensemble, ["ultrabert", "qwen"])
    b = select_candidates(hard, rel, ensemble, ["ultrabert", "qwen"])
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert np.all(a[0] != a[1])


def test_features_have_fixed_checkpoint_alignment_and_presence_flags():
    p, roster, hard, rel, ensemble = _synthetic_inputs()
    x, selected, alternative, names = build_features(
        p, ["N0001", "N0002", "N0003"], ["ultrabert", "qwen", "svc"],
        roster, hard, rel, ensemble)
    assert x.shape == (6, len(names))
    assert len(names) == len(set(names))
    absent = names.index("N0003::present")
    assert np.all(x[:, absent] == 0)
    assert np.all(selected != alternative)
    # Fixed layout means the same checkpoint slot is present across every row.
    assert np.all(x[:, names.index("N0001::present")] == 1)


def test_grouped_splits_keep_prompt_groups_disjoint():
    rows = np.arange(30)
    groups = np.repeat(np.arange(10), 3)
    for fit, held in GroupKFold(n_splits=5).split(rows, groups=groups):
        assert not (set(groups[fit]) & set(groups[held]))


def test_logistic_inference_is_reproducible_and_thresholded_pair_only():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 8)).astype(np.float32)
    y = (x[:, 0] + x[:, 2] > 0).astype(np.int8)
    scaler, model = _fit_logistic(x, y, 0.1)
    p1 = _probability(scaler, model, x[:12])
    p2 = _probability(scaler, model, x[:12])
    np.testing.assert_array_equal(p1, p2)
    base = np.zeros(12, dtype=np.int64)
    alt = np.ones(12, dtype=np.int64)
    pred = _apply_threshold(base, alt, p1, 0.5)
    assert set(pred).issubset({0, 1})
