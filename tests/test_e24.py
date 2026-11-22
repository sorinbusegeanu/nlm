import numpy as np

from src.e24 import MAX_CHECKPOINTS, _specialist_details, select_roster
from src.v2_checkpoint_trust import LABELS


def _ctx():
    rng = np.random.default_rng(23)
    n, cp, k = 72, 36, len(LABELS)
    y = np.arange(n) % k
    p = rng.random((cp, n, k))
    # Give each checkpoint a distinct subset of useful label predictions.
    for checkpoint in range(cp):
        rows = np.flatnonzero((y + checkpoint) % 5 == 0)
        p[checkpoint, rows, y[rows]] += 2.5
    p /= p.sum(axis=2, keepdims=True)
    ids = [f"N{i:04d}" for i in range(cp)]
    return {"checkpoint_ids": ids, "families": [f"family-{i % 6}" for i in range(cp)],
        "matrices": p.astype(np.float32), "y": y}


def test_e24_roster_is_limited_and_keeps_mandatory_e21_members():
    ctx = _ctx()
    candidates = ctx["checkpoint_ids"]
    roster, details = select_roster(ctx, np.arange(54), candidates,
                                   mandatory=["N0000", "N0001"])
    assert len(roster) == MAX_CHECKPOINTS
    assert len(set(roster)) == MAX_CHECKPOINTS
    assert {"N0000", "N0001"}.issubset(roster)
    assert details["train_rows"] == 54
    assert len(details["selection_steps"]) == MAX_CHECKPOINTS


def test_e24_selection_is_fold_local_and_deterministic():
    ctx = _ctx()
    train_rows = np.arange(54)
    candidates = ctx["checkpoint_ids"]
    roster_a, _ = select_roster(ctx, train_rows, candidates)
    roster_repeat, _ = select_roster(ctx, train_rows, candidates)
    altered = dict(ctx)
    altered["y"] = ctx["y"].copy()
    altered["y"][54:] = (altered["y"][54:] + 7) % len(LABELS)
    roster_b, _ = select_roster(altered, train_rows, candidates)
    assert roster_a == roster_repeat == roster_b


def test_e24_specialists_are_serialized_within_selected_roster():
    ctx = _ctx()
    pool = list(range(20))
    selected, details = _specialist_details(ctx, np.arange(54), pool)
    assert set(details) == set(LABELS)
    for label, roster in selected.items():
        assert len(roster) == 3
        assert set(roster).issubset(pool)
        assert len(details[LABELS[label]]) == 3

