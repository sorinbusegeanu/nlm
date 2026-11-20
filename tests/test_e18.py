import numpy as np

from src.e18 import _make_pairs, fit_specialist, specialist_scores, ordinary_scores


def test_pair_constructor_keeps_only_matched_prompt_sibling_pairs():
    labels = np.array(["a", "b", "a", "b", "a"])
    prompts = np.array(["p1", "p1", "p2", "p2", "p3"])
    pairs = _make_pairs(labels, prompts, ("a", "b"))
    left, right = pairs[("a", "b")]
    assert len(left) == 2
    assert all(prompts[a] == prompts[b] for a, b in zip(left, right))
    assert all(labels[a] == "a" and labels[b] == "b" for a, b in zip(left, right))


def test_contrastive_scores_are_available_per_response_without_prompt_ids():
    texts=[]; labels=[]; groups=[]
    for i in range(12):
        texts.extend([f"alpha style response {i} unique aa", f"beta style answer {i} unique bb"])
        labels.extend(["a", "b"]); groups.extend([f"p{i}", f"p{i}"])
    bundle=fit_specialist(texts,labels,groups,("a","b"),c=1.0)
    score=specialist_scores(bundle,["alpha style unseen response","beta style unseen answer"])
    ordinary=ordinary_scores(bundle,["alpha style unseen response","beta style unseen answer"])
    assert score.shape==(2,2) and ordinary.shape==(2,2)
    assert np.isfinite(score).all() and np.isfinite(ordinary).all()
    assert bundle["matched_pairs"]["a::b"]==12
