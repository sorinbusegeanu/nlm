import numpy as np

from src.e14_gate import (
    FEATURES,
    _make_pair_frame,
    _predictions,
)
from src.e14 import _validated_group_splits
from src.v2_checkpoint_trust import LABELS


def _toy_inputs(n_rows=30):
    truth = np.arange(n_rows) % len(LABELS)
    hard = np.stack([truth, (truth + 1) % len(LABELS), (truth + 1) % len(LABELS)])
    p = np.full((3, n_rows, len(LABELS)), 0.001 / (len(LABELS) - 1), dtype=float)
    p[np.arange(3)[:, None], np.arange(n_rows)[None, :], hard] = 0.999
    return p, hard, truth


def test_pair_frame_uses_distinct_labels_and_fold_local_statistics():
    p, hard, truth = _toy_inputs()
    target_rows = np.arange(18, 30)
    reliability = np.vstack([
        np.full(len(target_rows), 0.8),
        np.full(len(target_rows), 0.7),
        np.full(len(target_rows), 0.75),
    ])
    frame = _make_pair_frame(p, reliability, hard[:, target_rows], [0, 1, 2],
                             ["family_a", "family_a", "family_b"], truth,
                             np.arange(18), target_rows)

    assert len(frame) == len(target_rows)
    assert (frame._baseline_index != frame._competitor_index).all()
    assert set(FEATURES).issubset(frame.columns)
    assert frame.baseline_label.isin(LABELS).all()
    assert frame.competitor_label.isin(LABELS).all()
    assert frame._target_switch_corrects.eq(
        frame._competitor_index.to_numpy() == truth[target_rows]).all()
    assert np.isfinite(frame.loc[:, [c for c in FEATURES if c not in
                                     {"baseline_label", "competitor_label"}]].to_numpy()).all()


def test_intervention_changes_only_to_competitor_at_fixed_threshold():
    class FakeGate:
        def predict_proba(self, _):
            return np.asarray([[0.7, 0.3], [0.2, 0.8]])

    import pandas as pd
    frame = pd.DataFrame({"_baseline_index": [2, 3], "_competitor_index": [5, 1],
                          "baseline_label": [LABELS[2], LABELS[3]],
                          "competitor_label": [LABELS[5], LABELS[1]]})
    for feature in FEATURES:
        if feature not in frame:
            frame[feature] = 0.0
    pred, probability, switched = _predictions(FakeGate(), frame)
    assert pred.tolist() == [2, 1]
    assert switched.tolist() == [False, True]
    assert probability.tolist() == [0.3, 0.8]


def test_prompt_group_outer_folds_are_disjoint_and_cover_rows_once():
    rows = np.arange(50)
    truth = rows % len(LABELS)
    groups = np.asarray([f"prompt-{i // 2}" for i in rows])
    splits = _validated_group_splits(rows, truth, groups, 5)
    seen = np.zeros(len(rows), dtype=int)
    for fit, held in splits:
        assert not set(groups[rows[fit]]) & set(groups[rows[held]])
        seen[held] += 1
    assert np.all(seen == 1)
