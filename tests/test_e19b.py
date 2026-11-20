import json

import numpy as np
import pytest
import joblib

from src import e19b_group, e19b_reconstruct, e19b_evaluate


def test_reconstruction_json_and_cache_key_are_revision_scoped():
    parsed = e19b_reconstruct.parse_reconstruction(
        '{"reconstructed_prompt":"Compare A and B","topic":"comparison",'
        '"distinctive_details":["A","B"],"constraints":["table"]}')
    assert parsed["reconstructed_prompt"] == "Compare A and B"
    literal = e19b_reconstruct.parse_reconstruction(
        "{'reconstructed_prompt':'Describe the request','topic':'Hume's topic',"
        "'distinctive_details':['mention the 'black swan' counterexample',"
        "'the phrase \"language masks a mystery\"', 'literal {brace}'],"
        "'constraints':[]}")
    assert literal["topic"] == "Hume's topic"
    repaired=e19b_reconstruct.parse_reconstruction(
        '{"reconstructed_prompt":"Describe satellite fragments",'
        '"topic":"satellites","distinctive_details":["knives" (metal)],'
        '"constraints":[]}')
    assert 'knives" (metal)' in repaired["distinctive_details"][0]
    truncated=e19b_reconstruct.parse_reconstruction(
        '{"reconstructed_prompt":"Task","topic":"topic",'
        '"distinctive_details":["entity"],"constraints":["kept", "cut off')
    assert truncated["constraints"] == ["kept"]
    assert truncated["generation_repaired"] is True
    partial_details=e19b_reconstruct.parse_reconstruction(
        '{"reconstructed_prompt":"Task","topic":"topic",'
        '"distinctive_details":["a","b","cut')
    assert partial_details["distinctive_details"] == ["a","b"]
    assert partial_details["constraints"] == []
    pseudo=e19b_reconstruct.parse_reconstruction(
        "{'reconstructed_prompt':'Describe a fossil','topic':'Cambrian',"
        "'distinctive_details':['rotating nightmare' (describing Anomalocaris's mouth), 'limestone'],"
        "'constraints':[]}")
    assert "Anomalocaris's mouth" in pseudo["distinctive_details"][0]
    text = "response"
    assert e19b_reconstruct.cache_key(text, "revision-a") != e19b_reconstruct.cache_key(text, "revision-b")
    with pytest.raises(ValueError):
        e19b_reconstruct.parse_reconstruction("not JSON")


def test_cache_records_failures_and_success(tmp_path):
    cache = e19b_reconstruct.ReconstructionCache(tmp_path / "cache.sqlite")
    text = "response"
    key = e19b_reconstruct.cache_key(text, "rev")
    cache.save_attempt(key, text, "model", "rev", "v1", 1, error="transient")
    assert cache.entry(key)["status"] == "failed"
    value = {"reconstructed_prompt":"Task", "topic":"topic",
             "distinctive_details":[], "constraints":[]}
    cache.save_attempt(key, text, "model", "rev", "v1", 2, parsed=value, raw_output=json.dumps(value))
    assert cache.entry(key)["status"] == "ok"
    cache.close()


def test_complete_link_blocks_weak_transitive_bridge_and_is_order_invariant():
    ids = np.array(["a", "b", "c"])
    pairs = np.array([[0, 1], [1, 2], [0, 2]])
    scores = np.array([.95, .90, .20])
    got, _ = e19b_group.cluster_complete_link(ids, pairs, scores, threshold=.8)
    assert got[0] == got[1]
    assert got[2] != got[0]
    order = np.array([2, 0, 1])
    reordered_pairs = np.array([[1,2], [2,0], [1,0]])
    reordered, _ = e19b_group.cluster_complete_link(ids[order], reordered_pairs, scores, threshold=.8)
    by_id = dict(zip(ids[order], reordered))
    assert (by_id["a"] == by_id["b"]) and by_id["c"] != by_id["a"]


def test_cluster_size_limit_and_singletons():
    n = 20
    pairs = np.asarray([(a,b) for a in range(n) for b in range(a+1,n)])
    scores = np.ones(len(pairs))
    clusters, _ = e19b_group.cluster_complete_link(
        np.asarray([f"r{i:02}" for i in range(n)]), pairs, scores,
        threshold=.5, max_cluster_size=18)
    assert max(np.unique(clusters, return_counts=True)[1]) <= 18
    assert len(clusters) == n


def test_pair_features_are_finite_and_aligned():
    reconstructed = np.array(["write a short poem about mars", "summarize the mars mission"])
    responses = np.array(["Mars poem response", "Mars mission summary"])
    spaces = e19b_group.fit_similarity_spaces(reconstructed, responses, np.array([0,1]))
    semantic = np.array([[1.,0.], [.8,.6]], dtype=np.float32)
    details = [["Mars"], ["Mars mission"]]
    constraints = [["short poem"], ["summary"]]
    topics = ["Mars", "Mars"]
    pairs = np.array([[0,1]])
    features = e19b_group.pair_features(np.array([0,1]), pairs, spaces,
        reconstructed, responses, semantic, topics, details, constraints)
    assert features.shape == (1, len(e19b_group.FEATURE_NAMES))
    assert np.isfinite(features).all()


def test_candidate_retrieval_is_independent_of_input_row_order():
    reconstructed=np.array(["task alpha", "task alpha", "task beta", "task beta"])
    responses=np.array(["answer alpha", "answer alpha", "answer beta", "answer beta"])
    embeddings=np.eye(2,dtype=np.float32).repeat(2,axis=0)
    spaces=e19b_group.fit_similarity_spaces(reconstructed,responses,np.arange(4))
    ordered=e19b_group.candidate_pairs(np.arange(4),spaces,embeddings)
    shuffled=e19b_group.candidate_pairs(np.array([2,1,3,0]),spaces,embeddings)
    assert np.array_equal(ordered,shuffled)


def test_grouping_metrics_exact_pairwise_and_bcubed():
    result = e19b_evaluate.grouping_metrics(
        np.array(["a","a","b","b"]), np.array(["x","x","y","y"]))
    assert result["exact_groups"] == 2
    assert result["response_exact_grouping"] == 1
    assert result["pairwise_precision"] == result["pairwise_recall"] == result["pairwise_f1"] == 1
    assert result["bcubed_precision"] == result["bcubed_recall"] == result["bcubed_f1"] == 1
    assert result["predicted_size_distribution"] == {2: 2}
    assert result["group_size_analysis"]["2"]["reconstructed_groups"] == 2
    mixed = e19b_evaluate.grouping_metrics(
        np.array(["a","a","b","b"]), np.array(["x","x","x","y"]))
    assert mixed["incorrectly_merged_groups"] == 1
    assert mixed["split_original_groups"] == 1


def test_training_only_threshold_tuning_is_reproducible():
    n = 16
    rows = np.arange(n)
    groups = np.repeat([f"p{i}" for i in range(8)], 2)
    prompts = np.repeat([f"request about topic {i}" for i in range(8)], 2)
    responses = np.repeat([f"answer discussing topic {i}" for i in range(8)], 2)
    embeddings = np.eye(8, dtype=np.float32).repeat(2, axis=0)
    topics = groups.copy()
    details = [[f"topic {i}"] for i in np.repeat(np.arange(8), 2)]
    constraints = [["concise"] for _ in rows]
    first = e19b_evaluate.tune_on_training(rows, prompts, responses, embeddings,
        topics, details, constraints, groups)
    second = e19b_evaluate.tune_on_training(rows, prompts, responses, embeddings,
        topics, details, constraints, groups)
    assert first["C"] == second["C"]
    assert first["threshold"] == second["threshold"]


def test_unlabeled_inference_reuses_saved_grouping_bundle(tmp_path):
    n=8; ids=np.asarray([f"id{i}" for i in range(n)])
    groups=np.repeat(["g0","g1","g2","g3"],2)
    prompts=np.repeat([f"request topic {i}" for i in range(4)],2)
    responses=np.repeat([f"response topic {i}" for i in range(4)],2)
    topics=np.repeat([f"topic {i}" for i in range(4)],2)
    details=[[f"entity {i}"] for i in range(4) for _ in range(2)]
    constraints=[["brief"] for _ in range(n)]
    embeddings=np.eye(4,dtype=np.float32).repeat(2,axis=0)
    spaces=e19b_group.fit_similarity_spaces(prompts,responses,np.arange(n))
    pairs=e19b_group.candidate_pairs(np.arange(n),spaces,embeddings)
    features=e19b_group.pair_features(np.arange(n),pairs,spaces,prompts,responses,
        embeddings,topics,details,constraints)
    targets=(groups[pairs[:,0]]==groups[pairs[:,1]]).astype(np.int8)
    scorer=e19b_group.fit_pair_scorer(features,targets)
    bundle={"pair_scorer":scorer,"vectorizers":{name:spaces[name+"_vectorizer"] for name in
        ("reconstruction_char","reconstruction_word","response_char","response_word")},
        "threshold":.5}
    path=tmp_path/"group.joblib"; joblib.dump(bundle,path)
    reconstructions=[{"reconstructed_prompt":p,"topic":t,"distinctive_details":d,
                      "constraints":c} for p,t,d,c in zip(prompts,topics,details,constraints)]
    first=e19b_evaluate.predict_groups(ids,responses,reconstructions,embeddings,model_path=path)
    second=e19b_evaluate.predict_groups(ids,responses,reconstructions,embeddings,model_path=path)
    assert np.array_equal(first,second)
    permutation=np.array([6,2,4,0,7,1,5,3])
    shuffled=e19b_evaluate.predict_groups(ids[permutation],responses[permutation],
        [reconstructions[i] for i in permutation],embeddings[permutation],model_path=path)
    assert dict(zip(ids,first))==dict(zip(ids[permutation],shuffled))
