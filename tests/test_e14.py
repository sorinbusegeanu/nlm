import numpy as np
import pandas as pd

from src import e14
from src.e14 import (
    BASE_FEATURES,
    PAIR_FEATURES,
    _candidate_groups,
    _groups_from_records,
    _validated_group_splits,
    build_candidate_frame,
    select_candidate_scores,
)
from src.v2_checkpoint_trust import LABELS


def _probabilities(n_rows=30):
    n_labels = len(LABELS)
    truth = np.arange(n_rows) % n_labels
    hard = np.stack([truth, (truth + 1) % n_labels, (truth + 1) % n_labels])
    p = np.full((3, n_rows, n_labels), 0.001 / (n_labels - 1), dtype=np.float64)
    for checkpoint in range(3):
        p[checkpoint, np.arange(n_rows), hard[checkpoint]] = 0.999
    return p, hard, truth


def test_candidate_grouping_and_canonical_label_mapping():
    p, hard, truth = _probabilities()
    fit = np.arange(18)
    target = np.arange(18, 30)
    reliability = np.full((3, len(target)), 0.7)
    frame, _ = build_candidate_frame(
        p, reliability, hard[:, target], ["family_a", "family_a", "family_b"],
        truth, fit, target, include_pairwise=True)

    groups, y = _candidate_groups(frame)
    assert len(groups) == len(target)
    assert sum(groups) == len(frame) == len(y)
    assert frame.groupby("_row_index").candidate_label.nunique().eq(
        frame.groupby("_row_index").size()).all()
    assert set(frame.candidate_label).issubset(set(LABELS))
    assert frame.candidate_label.map({label: i for i, label in enumerate(LABELS)}).notna().all()
    assert set(BASE_FEATURES).issubset(frame.columns)
    assert set(PAIR_FEATURES).issubset(frame.columns)


def test_candidate_feature_alignment_and_pairwise_smoothing():
    p, hard, truth = _probabilities()
    fit = np.arange(18)
    target = np.arange(18, 30)
    reliability = np.linspace(0.2, 0.9, 3 * len(target)).reshape(3, len(target))
    frame, family_probabilities = build_candidate_frame(
        p, reliability, hard[:, target], ["family_a", "family_a", "family_b"],
        truth, fit, target, include_pairwise=True)

    assert family_probabilities.shape == (len(target), len(LABELS))
    assert np.allclose(family_probabilities.sum(axis=1), 1)
    assert frame.groupby("_row_index")._candidate_covered.first().all()
    assert frame.directional_candidate_to_competitor_rate.between(0, 1).all()
    assert frame.directional_competitor_to_candidate_rate.between(0, 1).all()
    assert frame.historical_candidate_win_probability_when_both_supported.between(0, 1).all()
    assert np.isfinite(frame.select_dtypes(include=["number"]).to_numpy()).all()


def test_prompt_group_crossfit_is_disjoint_and_covers_rows_once():
    rows = np.arange(48)
    truth = rows % len(LABELS)
    groups = np.asarray([f"prompt-{i // 2}" for i in rows])
    splits = _validated_group_splits(rows, truth, groups, 4)
    seen = np.zeros(len(rows), dtype=int)
    for fit, held in splits:
        assert not set(groups[rows[fit]]) & set(groups[rows[held]])
        seen[held] += 1
    assert np.all(seen == 1)


def test_selector_inference_is_deterministic_and_tie_breaks_by_family_reliability():
    frame = pd.DataFrame({
        "_row_index": [10, 10, 11, 11],
        "_label_index": [0, 1, 2, 3],
        "family_reliability_mean": [0.6, 0.8, 0.7, 0.7],
        "family_balanced_probability": [0.9, 0.2, 0.4, 0.6],
    })
    scores = np.asarray([0.5, 0.5, 0.8, 0.8])
    rows1, pred1 = select_candidate_scores(frame, scores)
    rows2, pred2 = select_candidate_scores(frame, scores)
    assert rows1.tolist() == rows2.tolist() == [10, 11]
    assert pred1.tolist() == pred2.tolist() == [1, 3]


def test_feature_builder_rejects_misaligned_hard_predictions():
    p, hard, truth = _probabilities()
    target = np.arange(18, 30)
    wrong = hard[:, target].copy()
    wrong[0, 0] = (wrong[0, 0] + 1) % len(LABELS)
    try:
        build_candidate_frame(p, np.full_like(wrong, 0.5, dtype=float), wrong,
                              ["family_a", "family_a", "family_b"], truth,
                              np.arange(18), target, include_pairwise=False)
    except ValueError as exc:
        assert "argmax" in str(exc)
    else:
        raise AssertionError("misaligned hard predictions were accepted")


def test_trained_selector_inference_is_reproducible(monkeypatch):
    monkeypatch.setitem(e14.CATBOOST_PARAMS, "iterations", 8)
    rows = []
    for row_index in range(50):
        correct = row_index % 2
        for candidate in range(2):
            record = {"_row_index": row_index, "_label_index": candidate,
                      "_is_correct": int(candidate == correct), "_candidate_count": 2,
                      "candidate_label": LABELS[candidate]}
            for feature_index, feature in enumerate(e14.BASE_FEATURES):
                if feature != "candidate_label":
                    record[feature] = float((row_index * (feature_index + 1) + candidate) % 17) / 17
            rows.append(record)
    frame = pd.DataFrame(rows)
    train = frame[frame._row_index < 40].copy()
    stop = frame[frame._row_index >= 40].copy()
    model, _, _, _ = e14._fit_selector("catboost", e14.BASE_FEATURES, train, stop)
    first = e14._predict_scores(model, "catboost", stop, e14.BASE_FEATURES)
    second = e14._predict_scores(model, "catboost", stop, e14.BASE_FEATURES)
    assert np.array_equal(first, second)
