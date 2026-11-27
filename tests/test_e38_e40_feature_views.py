import numpy as np
import json

from src.feature_factory import (EmpiricalSubwordStats, FoldFeatures, _distort_text,
                                 _shape_preserve, stylometry, tokenizer_fingerprint)


def test_baseline_stylometry_default_is_unchanged():
    texts=["Hello, WORLD!", "A second sentence."]
    old=stylometry(texts)
    new=stylometry(texts, {"baseline":["length","case","punctuation","function_words"],"groups":[]})
    np.testing.assert_array_equal(old,new)


def test_legacy_shape_view_configuration_is_preserved():
    texts=["same shape", "same shape", "different format"]
    features=FoldFeatures({"views":["shape"]})
    x=features.fit_transform(texts)
    vec=features.vectorizers['shape']
    assert vec.min_df==2 and vec.max_features is None and vec.lowercase is True
    assert features.transform(texts).shape==x.shape


def test_extended_svc_views_fit_and_transform_alignment():
    train=["Hello, WORLD 123!", "Hello there, world.", "A different response?", "Code: x = 3."]
    cfg={"views":["char_wb","word_punct","shape_preserve","distort"],
         "view_weights":{"char_wb":.85,"word_punct":1.15,"shape_preserve":.55,"distort":.75},
         "view_options":{v:{"min_df":1,"max_features":1000,"ngram_range":(2,3) if v in {"char_wb","shape_preserve"} else (1,2)} for v in ["char_wb","word_punct","shape_preserve","distort"]}}
    features=FoldFeatures(cfg); x=features.fit_transform(train); y=features.transform([train[0]])
    assert x.shape[0]==len(train) and y.shape==(1,x.shape[1])
    assert np.isfinite(x.data).all() and np.isfinite(y.data).all()


def test_shape_retains_full_word_lengths_and_distortion_masks_content_words():
    assert _shape_preserve("Ab9!")=="Xxd!"
    text=_distort_text("Hello useful words, and punctuation!")
    assert "<U5>" in text and "and" in text and "," in text and "!" in text


def test_tokenizer_fingerprint_feature_order_and_utf8_density():
    class Encoder:
        def encode(self,text,**kwargs): return list(range(max(1,len(text.split()))))
    class HfTokenizer:
        def encode(self,text,add_special_tokens=False): return list(range(max(1,len(text.split())+1)))
    x=tokenizer_fingerprint(["café model"],tokenizers=(Encoder(),Encoder(),HfTokenizer(),HfTokenizer()))
    assert x.shape==(1,16) and np.isfinite(x).all()
    assert x[0,-1] > 1.0


def test_empirical_stats_train_only_state_is_serializable_and_stable():
    train=["alpha beta alpha", "beta gamma alpha", "gamma beta"]
    state=EmpiricalSubwordStats().fit(train).state
    assert set(state)=={"tokenizer","unigram","bigram","alpha","interpolation"}
    model=EmpiricalSubwordStats(state)
    x=model.transform(["alpha beta", "unseen token"])
    assert x.shape==(2,20) and np.isfinite(x).all()
    np.testing.assert_array_equal(x,model.transform(["alpha beta", "unseen token"]))


def test_fold_features_fit_empirical_statistics_only_on_fit_rows():
    fit=["train alpha", "train beta"]
    held=["heldout unique token"]
    features=FoldFeatures({"views":["stylometry_raw"],"stylometry_recipe":{"features":["empirical_subword"]}})
    x=features.fit_transform(fit)
    state=features.stylometry_feature_state
    assert x.shape[0]==2 and state is not None
    import tiktoken
    held_ids=tiktoken.get_encoding('cl100k_base').encode(held[0])
    assert all(str(t) not in state['unigram'] for t in held_ids)
    assert features.transform(held).shape==(1,x.shape[1])


def test_stylometry_empirical_feature_order_and_width():
    train=["alpha beta alpha", "beta gamma alpha", "gamma beta"]
    stats=EmpiricalSubwordStats().fit(train).state
    x,names=stylometry(train,{"features":["empirical_subword"]},feature_state=stats,return_feature_names=True)
    assert x.shape==(len(train),len(names)) and len(names)==55
    assert names[-20]=="empirical_subword_00"


def test_empirical_state_checkpoint_serialization_and_inference(tmp_path):
    from src.model_factory import train_catboost_stylometry
    from src.inference import predict_catboost_stylometry
    texts=["alpha beta alpha", "beta gamma", "alpha delta", "gamma beta", "alpha epsilon", "beta zeta"]
    labels=["a","b","a","b","a","b"]
    stats=EmpiricalSubwordStats().fit(texts)
    recipe={"features":["empirical_subword"]}
    x,names=stylometry(texts,recipe,feature_state=stats.state,return_feature_names=True)
    val=["alpha beta", "gamma delta"]
    xv=stylometry(val,recipe,feature_state=stats.state)
    out=tmp_path/'checkpoint'
    train_catboost_stylometry(texts,labels,val,["a","b"],output_dir=out,
      iterations=3,depth=2,learning_rate=.1,l2_leaf_reg=3,thread_count=1,seed=19,
      validation_labels=["a","b"],early_stopping_rounds=2,feature_recipe=recipe,
      feature_matrices=(x,xv,names),feature_state=stats.state,checkpoint_selection_metric='accuracy')
    assert (out/'model.cbm').is_file() and (out/'feature_state.json').is_file()
    metadata=json.loads((out/'stylometry.json').read_text())
    assert metadata['feature_state_file']=='feature_state.json'
    probs,classes=predict_catboost_stylometry(out,val,["a","b"])
    assert probs.shape==(2,2) and np.isfinite(probs).all() and classes==["a","b"]
