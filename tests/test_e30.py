import numpy as np

from src.e13_b_inference import family_max_scores, make_features, route_family_max
from src.e28 import family_max_label_scores, finite_assignment_scores, route_family_max as e28_route
from src.e30_deploy import hungarian
from src.v2_checkpoint_trust import LABELS


def test_e13_b_shared_feature_and_route_order():
    rng = np.random.default_rng(17)
    p = rng.random((3, 12, len(LABELS)))
    p /= p.sum(axis=2, keepdims=True)
    members = [{"candidate_id": f"N{i:04d}", "family": f"family{i % 2}"}
               for i in range(3)]
    k = len(LABELS)
    stats = {"precision": np.full((3, k), .5), "recall": np.full((3, k), .5),
             "directional": np.full((3, k, k), 1 / k)}
    features = make_features(p, members, LABELS, stats)
    assert len(features) == 3 * 12
    assert features.checkpoint_id.iloc[:3].tolist() == ["N0000", "N0001", "N0002"]
    reliability = rng.random((3, 12))
    family_probs = p.mean(0)
    families = [x["family"] for x in members]
    pred = route_family_max(p.argmax(2), reliability, family_probs, families)
    _, shared_assignment = family_max_scores(p.argmax(2), reliability, family_probs, families)
    _, _, _, e28_assignment = e28_route(p.argmax(2), reliability, family_probs, families)
    assert np.allclose(shared_assignment, e28_assignment, rtol=0, atol=1e-12)
    assert np.array_equal(pred, e28_route(p.argmax(2), reliability, family_probs, families)[0])
    assert pred.shape == (12,)
    assert np.all((pred >= 0) & (pred < k))


def test_family_max_rejects_misaligned_inputs():
    p = np.full((2, 4, len(LABELS)), 1 / len(LABELS))
    stats = {"precision": np.full((2, len(LABELS)), .5),
             "recall": np.full((2, len(LABELS)), .5),
             "directional": np.full((2, len(LABELS), len(LABELS)), 1 / len(LABELS))}
    try:
        make_features(p, [{"candidate_id": "N1", "family": "a"}], LABELS, stats)
    except AssertionError:
        pass
    else:
        raise AssertionError("misaligned checkpoint roster should fail")


def test_shared_hungarian_stable_row_alignment():
    rng = np.random.default_rng(22)
    scores = rng.normal(size=(5, len(LABELS)))
    groups = np.asarray(["g1", "g1", "g2", "g2", "g2"])
    ids = np.asarray(["r0", "r1", "r2", "r3", "r4"])
    result = hungarian(scores, groups, ids)
    assert result.shape == (5,)
    assert len(set(result[:2].tolist())) == 2
    assert len(set(result[2:].tolist())) == 3
