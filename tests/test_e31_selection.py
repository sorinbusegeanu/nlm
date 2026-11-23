import numpy as np

from src.e31_selection import E31GreedySelector, _paired
from src.e30_deploy import hungarian as e30_hungarian
from src.e31_deploy import hungarian as e31_hungarian


def test_paired_corrections_and_regressions():
    got = _paired(np.array([0, 1, 1, 0]), np.array([1, 1, 0, 0]),
                  np.array([0, 0, 1, 0]))
    assert got == {"corrected": 2, "introduced": 1, "changed": 3, "net_correct": 1}


def test_e31_assignment_matches_shared_e30_runtime():
    rng = np.random.default_rng(20261123)
    scores = rng.normal(size=(8, 18))
    groups = np.array(["a", "a", "a", "b", "b", "c", "c", "c"])
    row_ids = np.array([f"r{i}" for i in range(len(groups))])
    assert np.array_equal(e31_hungarian(scores, groups, row_ids),
                          e30_hungarian(scores, groups, row_ids))


def test_consensus_selects_exactly_requested_additions_and_best_size():
    selector = E31GreedySelector.__new__(E31GreedySelector)
    selector.baseline_roster = [f"N{i:04d}" for i in range(17)]
    selector.candidates = [f"N{i:04d}" for i in range(100, 123)]
    selector.steps = 23
    selector.output_dir = __import__("pathlib").Path("/tmp/e31-selection-test")
    y = [0, 1, 0, 1]
    fold_predictions = {}
    for step in range(24):
        pred = y if step == 20 else [0, 0, 0, 0]
        fold_predictions[str(step)] = {"1": {"truth": y, "predictions": pred,
                                                    "row_ids": ["a", "b", "c", "d"]}}
    events = [{"checkpoint_id": sid, "outer_fold": 1, "step": i + 1,
               "marginal_balanced_accuracy": 0.01, "corrected": 1,
               "introduced": 0, "net_correct": 1}
              for i, sid in enumerate(selector.candidates)]
    candidate_evals = [{"checkpoint_id": sid, "marginal_balanced_accuracy": 0.01}
                       for sid in selector.candidates]
    result = selector.aggregate([], events, candidate_evals, fold_predictions)
    assert len(result["consensus_additions"]) == 23
    assert len(result["final_roster"]) == 40
    assert result["best_oof_roster_size_20_to_40"] == 37
