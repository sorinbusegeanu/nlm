import numpy as np
import pandas as pd
import os

from src import e27


def _fixture():
    labels = ["claude-fable-5", "claude-opus-4-6", "claude-opus-4-8",
              "claude-opus-5", "claude-sonnet-5", "gemini-3.1-pro-preview",
              "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
              "glm-5.3", "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna",
              "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra", "kimi-k3", "qwen3.8-max"]
    rows = []
    for p in range(60):
        for label in labels[:2 + p % 5]:
            rows.append({"id": f"r{p:03}-{label}", "text": f"response {p} {label}",
                         "model": label, "prompt_canon": f"prompt {p}"})
        if p == 0:
            rows.append({"id": "r000-extra", "text": "second response same label",
                         "model": labels[0], "prompt_canon": "prompt 0"})
    return pd.DataFrame(rows)


def test_prompt_split_is_deterministic_and_group_disjoint():
    frame = _fixture()
    a = e27.prompt_disjoint_split(frame)
    b = e27.prompt_disjoint_split(frame)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    prompt_sets = [set(frame.iloc[a[k]].prompt_canon) for k in a]
    assert not (prompt_sets[0] & prompt_sets[1] or prompt_sets[0] & prompt_sets[2]
                or prompt_sets[1] & prompt_sets[2])
    assert sorted(np.concatenate(list(a.values())).tolist()) == list(range(len(frame)))


def test_stratified_prompt_sample_reproducible_and_retains_all_sizes():
    frame=_fixture()
    a=e27.stratified_prompt_sample(frame,.8,seed=77)
    b=e27.stratified_prompt_sample(frame,.8,seed=77)
    assert a.id.tolist()==b.id.tolist()
    expected=set(frame.groupby("prompt_canon").model.nunique().unique())
    assert set(a.groupby("prompt_canon").model.nunique().unique())==expected
    splits=e27.prompt_disjoint_split(a,seed=77)
    for rows in splits.values():
        observed=set(a.iloc[rows].groupby("prompt_canon").model.nunique().unique())
        assert observed==expected


def test_tiny_sample_keeps_each_stratum_in_all_splits():
    frame=_fixture()
    sampled=e27.stratified_prompt_sample(frame,.01,seed=5)
    splits=e27.prompt_disjoint_split(sampled,seed=5)
    expected=set(frame.groupby("prompt_canon").model.nunique().unique())
    assert all(set(sampled.iloc[rows].groupby("prompt_canon").model.nunique().unique())==expected
               for rows in splits.values())
    prompt_sets=[set(sampled.iloc[rows].prompt_canon) for rows in splits.values()]
    assert not (prompt_sets[0]&prompt_sets[1] or prompt_sets[0]&prompt_sets[2] or prompt_sets[1]&prompt_sets[2])


def test_duplicate_rotation_is_one_per_label_and_covers_each_row_once():
    rotated, stats = e27.rotate_one_per_label(_fixture())
    assert not rotated.duplicated("id").any()
    assert rotated.groupby("view_group").model.nunique().eq(
        rotated.groupby("view_group").size()).all()
    assert stats["max_view_size"] <= 18
    assert set(rotated.id) == set(_fixture().id)


def test_hungarian_assignment_has_unique_labels_within_groups():
    scores = np.array([[.9, .8] + [0.] * 16, [.85, .7] + [0.] * 16,
                       [.1, .95] + [0.] * 16])
    pred = e27.evaluate_group_assignment(scores, ["a", "a", "b"], ["1", "2", "3"])
    assert len(set(pred[:2].tolist())) == 2
    assert pred[2] == 1


def test_scenario_selection_reproducible_and_unique_label_per_prompt():
    v0 = pd.DataFrame({"id": ["a", "b", "c", "d"],
                       "prompt_id": ["p", "p", "q", "q"],
                       "model": ["claude-fable-5", "claude-fable-5",
                                 "glm-5.3", "gemini-3.7-flash"]})
    one = e27.sample_incomplete_scenarios(v0, scenarios=4)
    two = e27.sample_incomplete_scenarios(v0, scenarios=4)
    assert all(np.array_equal(one[k], two[k]) for k in one)
    for selected in one.values():
        sub = v0.iloc[selected]
        assert not sub.duplicated(["prompt_id", "model"]).any()


def test_cuda_auto_check_does_not_count_its_own_context(monkeypatch):
    monkeypatch.setattr(e27.e19f.subprocess, "check_output",
                        lambda *a, **k: f"{os.getpid()}\n")
    assert e27.e19f._cuda_is_busy() is False
    monkeypatch.setattr(e27.e19f.subprocess, "check_output", lambda *a, **k: "999999\n")
    assert e27.e19f._cuda_is_busy() is True


def test_external_candidate_groupers_are_deterministic_and_complete_link():
    ids=np.array(["a","b","c","d"])
    pairs=np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
    cosine=np.array([.95,.1,.1,.1,.1,.94],dtype=np.float32)
    features=np.column_stack((cosine,np.zeros(6),np.zeros(6)))
    fa,_,_=e27._method_clusters("E19-F-A",ids,pairs,features,threshold=.8)
    fb,_,_=e27._method_clusters("E19-F-B",ids,pairs,features,threshold=.8,k=1)
    assert np.array_equal(fa, e27._method_clusters("E19-F-A",ids,pairs,features,threshold=.8)[0])
    assert fa[0]==fa[1] and fa[2]==fa[3] and fa[0]!=fa[2]
    assert fb[0]==fb[1] and fb[2]==fb[3] and fb[0]!=fb[2]


def test_confidence_gate_splits_weak_group_to_singletons():
    groups=np.array(["x","x","y"])
    gated=e27._confidence_gate(groups,{"x":.4,"y":1.},["a","b","c"],.5)
    assert gated[0]!=gated[1] and gated[2]=="y"
