import numpy as np
import pandas as pd
import pytest

from src.e20 import (_evidence, _group_audit, _score_with_reused_classifier,
                     _validate_e20b_e19b_group_alignment, group_feature_frame,
                     hungarian, load_reconstructed_groups)
from src.v2_checkpoint_trust import LABELS


def test_candidate_group_features_are_aligned_and_include_competition_signals():
    scores = np.zeros((3, len(LABELS)), dtype=float)
    scores[0, 0], scores[0, 1] = .9, .2
    scores[1, 0], scores[1, 1] = .8, .3
    scores[2, 2], scores[2, 3] = .8, .4
    groups = np.array(["p1", "p1", "p2"])
    frame = group_feature_frame(scores, groups, np.array(["a", "b", "c"]),
                                np.array(["same answer style", "similar answer style", "other text"]),
                                oracle=True)
    assert len(frame) == 3 * len(LABELS)
    assert frame.candidate_label.iloc[:len(LABELS)].tolist() == list(LABELS)
    assert frame.loc[0, "group_size"] == 2
    assert frame.loc[0, "group_candidate_best_other"] == .8
    assert np.isclose(frame.loc[0, "group_candidate_margin_other"], .1)
    assert frame.loc[2 * len(LABELS), "group_size"] == 1
    assert np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()


def test_hungarian_assignment_aligns_rows_and_enforces_unique_labels():
    scores = np.zeros((3, len(LABELS)), dtype=float)
    scores[0, 0], scores[0, 1] = .9, .8
    scores[1, 0], scores[1, 1] = .85, .1
    scores[2, 0] = .7
    pred = hungarian(scores, np.array(["p", "p", "singleton"]), ["a", "b", "c"])
    assert len(set(pred[:2].tolist())) == 2
    assert pred[2] == 0


def test_group_audit_counts_exact_split_and_merge_cases():
    audit = _group_audit(np.array(["p", "p", "q", "q"]),
                         np.array(["a", "b", "c", "c"]))
    assert audit["split_groups"] == 1
    assert audit["merged_groups"] == 0
    assert audit["exact_groups"] == 1


def test_e13_reliability_feature_fit_rejects_prompt_overlap():
    ctx={"groups":np.array(["prompt-a","prompt-a"])}
    with pytest.raises(ValueError,match="overlap"):
        _evidence(ctx,np.array([0]),np.array([1]))


def test_reconstructed_group_loader_drops_prompt_targets(tmp_path):
    base=tmp_path/"artifacts/analysis/e19b";base.mkdir(parents=True)
    pd.DataFrame({"id":["r1","r2"],"true_prompt_id":["p1","p2"],
                  "predicted_group_id":["g1","g1"],"fold":[1,1]}).to_parquet(base/"oof_group_predictions.parquet",index=False)
    pd.DataFrame({"left_id":["r1"],"right_id":["r2"],"similarity":[.93],
                  "same_true_group":[False]}).to_parquet(base/"oof_pair_diagnostics.parquet",index=False)
    loaded=load_reconstructed_groups(tmp_path,required=True)
    assert "true_prompt_id" not in loaded.columns
    assert "same_true_group" not in loaded.attrs["pair_scores"]
    assert loaded.e20_group.nunique()==1


def test_reconstructed_group_features_use_edge_confidence_not_oracle_constants():
    scores=np.zeros((2,len(LABELS)),dtype=float)
    scores[0,0]=.8;scores[1,0]=.7
    frame=group_feature_frame(scores,np.array(["g","g"]),np.array(["r1","r2"]),
        np.array(["shared response text","shared response text"]),
        pair_scores={("r1","r2"):.93},oracle=False)
    assert frame.loc[0,"group_size"]==2
    assert np.isclose(frame.loc[0,"group_confidence"],.93)
    assert np.isclose(frame.loc[0,"group_edge_mean"],.93)


def test_e20d_fold_and_group_alignment_matches_e20b_memberships():
    ctx={"row_ids":np.array(["r1","r2","r3","r4"]),
         "outer_folds":[{"fold":1,"held_rows":np.array([0,1])},
                        {"fold":2,"held_rows":np.array([2,3])}]}
    groups=pd.DataFrame({"id":["r1","r2","r3","r4"],
        "e20_group":["1::a","1::a","2::b","2::c"]})
    predictions=pd.DataFrame({"id":["r1","r2","r3","r4"],"fold":[1,1,2,2],
        "predicted_group_id":["outer1::x","outer1::x","outer2::y","outer2::z"]})
    audit=_validate_e20b_e19b_group_alignment(ctx,groups,predictions)
    assert [x["adjusted_rand_index"] for x in audit]==[1.0,1.0]
    predictions.loc[3,"predicted_group_id"]="outer2::y"
    with pytest.raises(ValueError,match="differ"):
        _validate_e20b_e19b_group_alignment(ctx,groups,predictions)


def test_e20d_scores_with_saved_classifier_and_aligns_feature_names():
    class SavedClassifier:
        feature_names_=["candidate_label","score"]
        def predict_proba(self,frame):
            assert list(frame.columns)==self.feature_names_
            values=frame.score.to_numpy(dtype=float)
            return np.column_stack([1-values,values])
    feature_frame=pd.DataFrame({"score":np.linspace(.1,.9,len(LABELS)),
        "candidate_label":LABELS})
    matrix=_score_with_reused_classifier(SavedClassifier(),feature_frame)
    assert matrix.shape==(1,len(LABELS))
    assert np.allclose(matrix[0],np.linspace(.1,.9,len(LABELS)))
