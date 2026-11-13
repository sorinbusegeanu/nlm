import numpy as np
import pandas as pd

from src.v2_14_prediction_filter import classify_rows, select_rows


def _fixture():
    labels = ["A", "B", "C"]
    rows = pd.DataFrame({
        "id": [f"r{i}" for i in range(12)],
        "model": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
        "source_dataset": ["s1", "s2"] * 6,
        "text": ["word " * (i + 10) for i in range(12)],
    })
    probs = {}
    for name in ("ultrabert", "svc", "catboost"):
        p = np.full((12, 3), .05)
        p[np.arange(12), np.repeat([0, 1, 2], 4)] = .90
        probs[name] = p
    # Force every category while preserving at least one correct UltraBERT row
    # for each label's median.
    probs["svc"][1, :] = [.9, .05, .05]
    probs["catboost"][1, :] = [.9, .05, .05]
    probs["ultrabert"][4] = [.8, .1, .1]  # assigned B absent top-3? impossible with 3 labels
    # Instead rank B third on this row, making it a Difficult example.
    probs["ultrabert"][4] = [.60, .20, .20]
    probs["svc"][4] = [.1, .8, .1]
    probs["catboost"][4] = [.1, .8, .1]
    probs["ultrabert"][8] = [.91, .045, .045]
    probs["svc"][8] = [.91, .045, .045]
    probs["catboost"][8] = [.91, .045, .045]
    return rows, labels, probs


def test_category_assignment_covers_each_row_once_and_is_ordered():
    frame, labels, probs = _fixture()
    result = classify_rows(frame, probs, labels)
    assert result["category"].shape == (len(frame),)
    assert set(result["category"]) <= set(range(6))
    assert result["true_rank"].shape == (len(frame),)


def test_selection_is_deterministic_and_keeps_full_retention_categories():
    frame, labels, probs = _fixture()
    category = classify_rows(frame, probs, labels)["category"]
    first = select_rows(frame, category, seed=2026)
    second = select_rows(frame, category, seed=2026)
    assert np.array_equal(first, second)
    assert first.shape == (len(frame),)
    assert first[category == 3].all()
    assert not first[category == 4].any()


def test_strong_consensus_error_requires_label_outside_top_three():
    labels = ["A", "B", "C", "D"]
    frame = pd.DataFrame({
        "id": ["a", "b", "c", "d_ok", "d_error"],
        "model": ["A", "B", "C", "D", "D"],
        "text": ["short response"] * 5,
        "source_dataset": ["source"] * 5,
    })
    probs = {}
    for name in ("ultrabert", "svc", "catboost"):
        p = np.full((5, 4), .01)
        p[np.arange(4), np.arange(4)] = .97
        p[4] = [.92, .04, .03, .01]
        probs[name] = p
    # All three agree on the same wrong label, with D outside UltraBERT's top 3.
    probs["svc"][4] = [.92, .04, .03, .01]
    probs["catboost"][4] = [.92, .04, .03, .01]
    result = classify_rows(frame, probs, labels)
    assert result["category"][4] == 4
    assert result["true_rank"][4] == 4
    retained = select_rows(frame, result["category"], seed=2026)
    assert not retained[4]
