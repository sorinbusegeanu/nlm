import numpy as np
import pandas as pd

from src import e13_reliability_router as e13
from src.e30_selection import E30GreedySelector
from src.v2_checkpoint_trust import LABELS


class _ConstantRouter:
    def fit(self, x, y, **kwargs):
        self.p = float(np.mean(y))
        return self

    def predict_proba(self, x):
        return np.column_stack([np.full(len(x), 1 - self.p), np.full(len(x), self.p)])


def _fixture(tmp_path):
    rng = np.random.default_rng(32)
    rows = []
    for group in range(8):
        for label in LABELS:
            rows.append({"id": f"r{group:02d}-{label}", "prompt_id": f"p{group:02d}",
                         "text": f"answer {group} {label}", "model": label})
    v0 = pd.DataFrame(rows)
    probabilities = {}
    members = {}
    for i, sid in enumerate(("N0001", "N0002", "N0003", "N0004")):
        raw = rng.random((len(v0), len(LABELS)))
        raw[np.arange(len(v0)), np.arange(len(v0)) % len(LABELS)] += 1
        raw /= raw.sum(1, keepdims=True)
        probabilities[sid] = raw
        members[sid] = {"model": "svc" if i % 2 == 0 else "catboost_stylometry"}
    selector = E30GreedySelector(v0, members, probabilities, ["N0001", "N0002"],
                                  tmp_path, outer_folds=5, inner_folds=4, steps=2, iterations=2)
    return selector


def test_nested_router_features_keep_prompt_groups_disjoint(tmp_path, monkeypatch):
    selector = _fixture(tmp_path)
    original = e13.build_feature_frame
    calls = []

    def checked(probabilities, ids, families, truth_fit, fit_rows, target_rows, **kwargs):
        assert set(fit_rows).isdisjoint(set(target_rows))
        assert set(selector.groups[np.asarray(fit_rows)]).isdisjoint(
            set(selector.groups[np.asarray(target_rows)]))
        calls.append((len(fit_rows), len(target_rows)))
        return original(probabilities, ids, families, truth_fit, fit_rows, target_rows, **kwargs)

    monkeypatch.setattr(e13, "build_feature_frame", checked)
    monkeypatch.setattr(e13, "_new_catboost", lambda iterations=None: _ConstantRouter())
    outer_train = np.arange(len(selector.v0) - 18)
    metrics, pred = selector.evaluate_roster(1, outer_train, selector.baseline_roster)
    assert pred.shape == (len(outer_train),)
    assert 0 <= metrics["balanced_accuracy"] <= 1
    assert len(calls) == 20  # 4 held folds x (4 OOF feature folds + target features)
    selector.evaluate_roster(1, outer_train, selector.baseline_roster)
    assert len(calls) == 20  # cached, aligned replay


def test_aggregate_uses_frequency_rank_gain_and_positive_gate(tmp_path):
    selector = _fixture(tmp_path)
    events = [
        {"outer_fold": 1, "step": 1, "checkpoint_id": "N0003", "marginal_balanced_accuracy": .02},
        {"outer_fold": 2, "step": 2, "checkpoint_id": "N0003", "marginal_balanced_accuracy": .01},
        {"outer_fold": 3, "step": 1, "checkpoint_id": "N0004", "marginal_balanced_accuracy": .03},
        {"outer_fold": 4, "step": 2, "checkpoint_id": "N0004", "marginal_balanced_accuracy": -.04},
    ]
    gains = [{"checkpoint_id": "N0003", "marginal_balanced_accuracy": .01},
             {"checkpoint_id": "N0004", "marginal_balanced_accuracy": -.01}]
    result = selector.aggregate([], events, gains, {})
    assert result["candidate_ranking"][0]["checkpoint_id"] == "N0003"
    assert result["final_roster"] == ["N0001", "N0002", "N0003"]
    assert result["qualified_additions"] == ["N0003"]
