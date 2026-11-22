import numpy as np
from src.e22 import build_features, select_specialists
from src.e20 import hungarian
from src.v2_checkpoint_trust import LABELS


def _ctx():
    rng = np.random.default_rng(41)
    n, cp, k = 36, 6, len(LABELS)
    y = np.arange(n) % k
    probs = rng.random((cp, n, k))
    probs /= probs.sum(axis=2, keepdims=True)
    # Give each label a distinct useful specialist, with noise preserved.
    for row, label in enumerate(y):
        probs[label % cp, row, label] += 2.0
    probs /= probs.sum(axis=2, keepdims=True)
    return {"matrices": probs.astype(np.float32), "checkpoint_ids": [f"N{i:04d}" for i in range(cp)],
        "families": [f"family-{i % 3}" for i in range(cp)], "y": y,
        "row_ids": np.asarray([f"r{i}" for i in range(n)])}


def test_e22_feature_alignment_is_row_major_and_canonical():
    ctx = _ctx()
    rows = np.asarray([7, 2, 11])
    frame = build_features(ctx, rows, list(range(6)))
    assert len(frame) == len(rows) * len(LABELS)
    assert frame.candidate_label.iloc[:len(LABELS)].tolist() == list(LABELS)
    # Candidate probability values map to the supplied row order.
    for i, row in enumerate(rows):
        block = frame.iloc[i * len(LABELS):(i + 1) * len(LABELS)]
        assert block.candidate_label.tolist() == list(LABELS)
        for cp, sid in enumerate(ctx["checkpoint_ids"]):
            assert np.allclose(block[f"cp::{sid}::prob"].to_numpy(),
                               ctx["matrices"][cp, row, :])


def test_specialist_selection_uses_only_supplied_training_rows():
    ctx = _ctx()
    pool = list(range(6))
    train = np.arange(24)
    picked_a, details_a = select_specialists(ctx, train, pool)
    altered = dict(ctx)
    altered["y"] = ctx["y"].copy()
    altered["y"][24:] = (altered["y"][24:] + 5) % len(LABELS)
    picked_b, details_b = select_specialists(altered, train, pool)
    assert picked_a == picked_b
    assert details_a == details_b
    assert all(len(v) <= 3 for v in picked_a.values())


def test_hungarian_assignment_enforces_unique_labels_and_covers_rows():
    scores = np.asarray([[.9, .8, .1] + [.0] * (len(LABELS)-3),
                         [.95, .2, .1] + [.0] * (len(LABELS)-3),
                         [.1, .9, .2] + [.0] * (len(LABELS)-3)])
    pred = hungarian(scores, np.asarray(["g", "g", "g"]), ["a", "b", "c"])
    assert len(pred) == 3
    assert len(set(pred.tolist())) == 3

