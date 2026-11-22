import numpy as np
import pandas as pd

from src.e26 import (build_external_random_groups, build_v0_group,
                     evidence_features, reconstruct_groups, fit_text_grouper,
                     validate_external_groups, validate_v0_group,
                     oof_reconstructed_groups)
from src.v2_checkpoint_trust import LABELS
from src.e20 import hungarian


def _v0(n_prompts=50):
    rows=[]
    for p in range(n_prompts):
        for model in LABELS:
            rows.append({"id":f"v0-{p}-{model}", "prompt_id":f"p{p}",
                         "text":f"Response {p} from {model}. A distinct explanation.",
                         "model":model})
    return pd.DataFrame(rows)


def test_v0_group_scenarios_cover_sizes_and_are_reproducible():
    source=_v0(150)
    a=build_v0_group(source,n_scenarios=1,target_rows=500,seed=2026)
    b=build_v0_group(source,n_scenarios=1,target_rows=500,seed=2026)
    validate_v0_group(a,n_scenarios=1,target_rows=500)
    pd.testing.assert_frame_equal(a,b)
    assert set(a.intended_group_size)==set(range(1,19))
    assert a.groupby("group_id").model.nunique().equals(a.groupby("group_id").size())


def test_external_random_groups_keep_source_prompt_and_unique_label():
    v0=_v0()
    rows=[]
    for p in range(5):
        for model in LABELS[:min(4+p, 8)]:
            rows.append({"id":f"e-{p}-{model}","text":f"External response {p} by {model}.",
                "model":model,"source_dataset":"source-a","source_prompt_id":f"s{p}",
                "source_file":"synthetic.parquet"})
    result,audit=build_external_random_groups(pd.DataFrame(rows),v0,seed=7)
    validate_external_groups(result,v0)
    assert result.original_prompt_id.nunique()==result.group_id.nunique()
    assert not result.duplicated("original_id").any()
    assert audit["v0_normalized_text_overlap"]==0


def test_reconstructed_groups_and_candidate_feature_rebuild_alignment():
    rows=[]
    for p in range(6):
        for model in LABELS[:3]:
            rows.append({"original_id":f"{p}-{model}","original_prompt_id":f"p{p}",
                         "text":f"Explain item {p} clearly. {model} response."})
    frame=pd.DataFrame(rows)
    grouper=fit_text_grouper(frame,seed=9)
    g,confidence,_=reconstruct_groups(grouper,frame.text,frame.original_id)
    assert len(g)==len(frame)==len(confidence)
    probs=np.full((len(frame),3,len(LABELS)),1/len(LABELS),dtype=np.float32)
    x=evidence_features(probs,np.asarray([LABELS[0]]*len(frame)),groups=g,
                        group_conf=confidence,include_group=True)
    assert len(x)==len(frame)*len(LABELS)
    assert set(["group_size","group_confidence","candidate_label"]).issubset(x.columns)


def test_oof_group_reconstruction_is_prompt_partitioned():
    rows=[]
    for p in range(8):
        for model in LABELS[:3]:
            rows.append({"original_id":f"{p}-{model}","original_prompt_id":f"p{p}",
                         "text":f"Explain item {p} clearly. {model} response."})
    frame=pd.DataFrame(rows)
    groups,confidence=oof_reconstructed_groups(frame,n_splits=4,seed=17)
    assert len(groups)==len(frame)==len(confidence)
    assert np.isfinite(confidence).all()
    for prompt in frame.original_prompt_id.unique():
        rows_for_prompt=np.flatnonzero(frame.original_prompt_id.to_numpy()==prompt)
        assert len({x.split("::",1)[0] for x in groups[rows_for_prompt]})==1


def test_hungarian_respects_group_boundaries_and_unique_assignments():
    scores=np.zeros((3,len(LABELS)),dtype=np.float32)
    scores[0,0]=.9; scores[0,1]=.8
    scores[1,0]=.95; scores[1,1]=.1
    scores[2,2]=.7
    pred=hungarian(scores,np.asarray(["g1","g1","g2"]))
    assert len(set(pred[:2].tolist()))==2
    assert pred[2]==2
