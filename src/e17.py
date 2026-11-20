"""E17: Unicode-preserving model typography fingerprints and E13 reranking."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import e16_common as common
from .v2_checkpoint_trust import LABELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/e17"
SEED = 2026
THREADS = 24
INNER_FOLDS = 3
CS = (0.01, 0.1, 1.0, 10.0)
WEIGHTS = (0.25, 0.5, 1.0, 2.0, 4.0)
MARGINS = (0.0, 0.05, 0.10, 0.20, 0.35)


def _add(counts, name, value, denominator):
    counts[name + "::present"] = float(value > 0)
    counts[name + "::per_char"] = float(value / max(1, denominator) * 1000.0)


def typography_features(text: str) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Extract explicit counts/presence/rates without normalizing raw Unicode."""
    t = str(text)
    n = max(1, len(t)); lines = t.splitlines() or [t]
    words = re.findall(r"\b[^\W\d_]+\b", t, flags=re.UNICODE)
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", t) if x.strip()]
    groups: dict[str, dict[str, float]] = {x: {} for x in (
        "unicode_punctuation", "markdown", "whitespace", "punctuation",
        "sentence_structure", "capitalization", "unusual_unicode")}
    def counts(group, prefix, mapping):
        for name, value in mapping.items():
            _add(groups[group], prefix + name, value, n)
    counts("unicode_punctuation", "unicode_", {
        "em_dash": t.count("—"), "en_dash": t.count("–"),
        "left_curly_double_quote": t.count("“"), "right_curly_double_quote": t.count("”"),
        "left_curly_single_quote": t.count("‘"), "right_curly_single_quote": t.count("’"),
        "straight_double_quote": t.count('"'), "straight_apostrophe": t.count("'"),
        "unicode_ellipsis": t.count("…"), "three_dot_ellipsis": len(re.findall(r"(?<!\.)\.{3}(?!\.)", t)),
    })
    markdown_map = {
        "heading_lines": sum(bool(re.match(r"^\s{0,3}#{1,6}(?:\s|$)", x)) for x in lines),
        "bullet_lines": sum(bool(re.match(r"^\s*[-+*]\s+", x)) for x in lines),
        "numbered_lines": sum(bool(re.match(r"^\s*\d{1,3}[.)]\s+", x)) for x in lines),
        "code_fence_lines": sum("```" in x or "~~~" in x for x in lines),
        "bold_markers": len(re.findall(r"\*\*|__", t)),
        "italic_markers": len(re.findall(r"(?<!\*)\*(?!\*)|(?<!_)_(?!_)", t)),
        "inline_code_markers": t.count("`"),
        "blockquote_lines": sum(bool(re.match(r"^\s*>\s?", x)) for x in lines),
        "link_syntax": len(re.findall(r"\[[^\]]+\]\([^)]*\)", t)),
    }
    counts("markdown", "markdown_", markdown_map)
    blank_lines = sum(not x.strip() for x in lines)
    indent_lines = sum(bool(re.match(r"^[ \t]+\S", x)) for x in lines)
    trailing = sum(x.endswith((" ", "\t")) for x in lines)
    counts("whitespace", "space_", {
        "spaces": t.count(" "), "tabs": t.count("\t"), "lf_endings": t.count("\n"),
        "crlf_endings": t.count("\r\n"), "cr_endings": t.count("\r")-t.count("\r\n"),
        "paragraph_breaks": len(re.findall(r"\n\s*\n", t)), "blank_lines": blank_lines,
        "indented_lines": indent_lines, "trailing_space_lines": trailing,
        "line_count": len(lines), "max_line_chars": max(map(len, lines)),
    })
    for key, value in {"space_ratio": t.count(" ")/n, "tab_ratio": t.count("\t")/n,
                       "newline_ratio": t.count("\n")/n,
                       "indent_line_ratio": indent_lines/max(1,len(lines)),
                       "blank_line_ratio": blank_lines/max(1,len(lines))}.items():
        groups["whitespace"][key] = float(value)
        groups["whitespace"][key+"::present"] = float(value > 0)
    punct = ".,;:!?-—–()[]{}\"'“”‘’…/\\"
    for ch in punct:
        name = f"char_{ord(ch):04x}"
        _add(groups["punctuation"], "punct_"+name, t.count(ch), n)
    for label, pattern in {
        "exclamation": r"!+", "question": r"\?+", "semicolon": r";", "colon": r":",
        "comma": r",", "period": r"\.", "parenthetical": r"\([^)]*\)",
    }.items():
        v = len(re.findall(pattern, t))
        _add(groups["punctuation"], "punct_pattern_"+label, v, n)
    sentence_words = np.asarray([len(re.findall(r"\b\w+\b", s, flags=re.UNICODE)) for s in sentences], dtype=float)
    if not len(sentence_words): sentence_words = np.asarray([0.0])
    for name, value in {"sentence_count": len(sentences), "mean_sentence_words": float(sentence_words.mean()),
                        "std_sentence_words": float(sentence_words.std()),
                        "median_sentence_words": float(np.median(sentence_words)),
                        "max_sentence_words": float(sentence_words.max()),
                        "mean_word_chars": float(np.mean([len(w) for w in words])) if words else 0.0,
                        "word_count": len(words)}.items():
        groups["sentence_structure"][name] = float(value)
        groups["sentence_structure"][name+"::per_100_words"] = float(value/max(1,len(words))*100)
        groups["sentence_structure"][name+"::present"] = float(value > 0)
    upper = sum(c.isupper() for c in t); letters = sum(c.isalpha() for c in t)
    caps = sum(bool(re.fullmatch(r"[A-Z]{2,}", w)) for w in re.findall(r"\b\w+\b", t))
    counts("capitalization", "case_", {"uppercase_chars":upper,"alphabetic_chars":letters,
        "all_caps_words":caps,"titlecase_words":sum(w.istitle() for w in words),
        "lowercase_chars":sum(c.islower() for c in t)})
    groups["capitalization"]["uppercase_ratio"] = upper/max(1,letters)
    groups["capitalization"]["all_caps_word_ratio"] = caps/max(1,len(words))
    cats=Counter(unicodedata.category(ch) for ch in t)
    nonascii=sum(ord(ch)>127 for ch in t)
    unusual=sum(ord(ch)>127 and not unicodedata.category(ch).startswith("P") for ch in t)
    counts("unusual_unicode", "unicode_extra_", {"non_ascii":nonascii,"non_ascii_nonpunct":unusual,
        "combining_marks":sum(unicodedata.category(ch).startswith("M") for ch in t),
        "format_chars":cats["Cf"],"symbol_chars":sum(v for k,v in cats.items() if k.startswith("S")),
        "replacement_chars":t.count("\ufffd"),"zero_width_chars":sum(ch in "\u200b\u200c\u200d\ufeff" for ch in t)})
    flat={}
    feature_groups={}
    for group, values in groups.items():
        for name, value in values.items():
            flat[name]=float(value); feature_groups[name]=group
    if not np.isfinite(np.asarray(list(flat.values()),dtype=float)).all():
        raise ValueError("Typography features contain non-finite values")
    return flat, feature_groups


def feature_matrix(texts):
    records=[]; names=None; owner=None
    for text in texts:
        row, group=typography_features(text)
        if names is None: names=sorted(row); owner=group
        elif sorted(row)!=names: raise ValueError("Typography feature schema changed between rows")
        records.append([row[name] for name in names])
    return np.asarray(records,dtype=np.float32),names,owner


def _fit(X, y, c):
    scaler=StandardScaler(); z=scaler.fit_transform(X)
    clf=LogisticRegression(C=float(c),solver="lbfgs",max_iter=2000,random_state=SEED)
    clf.fit(z,y)
    return scaler,clf


def _predict(scaler,clf,X):
    p=clf.predict_proba(scaler.transform(X)); out=np.zeros((len(X),len(LABELS)),dtype=np.float64)
    for j,label_i in enumerate(clf.classes_): out[:,int(label_i)]=p[:,j]
    return out


def _rerank(y, base, alt, e13_scores, fp_probs, weight, margin):
    rows=np.arange(len(y)); scores=common.finite_scores(e13_scores)
    e13_gap=scores[rows,base]-scores[rows,alt]
    fp_evidence=fp_probs[rows,alt]-fp_probs[rows,base]
    switch=(float(weight)*fp_evidence-e13_gap)>float(margin)
    pred=np.where(switch,alt,base).astype(np.int64)
    return pred,switch


def _tune_outer(ctx,fit_rows,X,names,owners):
    y=ctx["y"];groups=ctx["groups"]
    records=[]
    for inner,(fi,vi) in enumerate(GroupKFold(INNER_FOLDS).split(fit_rows,y[fit_rows],groups[fit_rows]),1):
        train,valid=fit_rows[fi],fit_rows[vi]
        e13=common.fit_e13_predict(ctx,train,valid)
        row={"valid":valid,"base":e13["baseline"],"alt":e13["alternative"],"scores":e13["label_scores"],"probs":{}}
        for c in CS:
            scaler,clf=_fit(X[train],y[train],c)
            row["probs"][c]=_predict(scaler,clf,X[valid])
        records.append(row)
        print(f"E17 nested fold {inner}/{INNER_FOLDS} feature fit done",flush=True)
    selections={}
    # Standalone C is selected on inner group-OOF rows.
    best=None
    for c in CS:
        yy=np.concatenate([y[r["valid"]] for r in records]); pp=np.concatenate([r["probs"][c] for r in records]); pred=pp.argmax(1)
        ba=float(balanced_accuracy_score(yy,pred));acc=float(accuracy_score(yy,pred));key=(ba,acc,-c)
        if best is None or key>best[0]:best=(key,c)
    selections["standalone_C"]=best[1]
    # Joint selection is based entirely on nested prompt-group OOF outputs.
    best=None
    for c in CS:
        for w in WEIGHTS:
            for margin in MARGINS:
                yall=[]; pall=[]
                for r in records:
                    p,_=_rerank(y[r["valid"]],r["base"],r["alt"],r["scores"],r["probs"][c],w,margin)
                    yall.extend(y[r["valid"]].tolist());pall.extend(p.tolist())
                ba=float(balanced_accuracy_score(yall,pall));acc=float(accuracy_score(yall,pall))
                key=(ba,acc,-margin,-w,-c)
                if best is None or key>best[0]:best=(key,{"C":c,"weight":w,"margin":margin,"tuning_balanced_accuracy":ba,"tuning_accuracy":acc})
    selections["combined"]=best[1]
    # Fold-local feature ablation C selection: all and leave-one-category-out.
    category_names=sorted(set(owners.values()))
    subset_names={"all":names,**{f"without_{g}":[n for n in names if owners[n]!=g] for g in category_names},
                  **{f"only_{g}":[n for n in names if owners[n]==g] for g in category_names}}
    subset_idx={key:np.asarray([names.index(n) for n in value],dtype=np.int64) for key,value in subset_names.items()}
    ablation={}
    for key,ix in subset_idx.items():
        candidates=[]
        for c in CS:
            yall=[];pall=[]
            # Select each ablation's C with the same group folds, fitting its
            # scaler and classifier on only the corresponding inner training rows.
            for fi,vi in GroupKFold(INNER_FOLDS).split(fit_rows,y[fit_rows],groups[fit_rows]):
                tr,va=fit_rows[fi],fit_rows[vi]
                sc,cl=_fit(X[tr][:,ix],y[tr],c)
                p=_predict(sc,cl,X[va][:,ix]);yall.extend(y[va].tolist());pall.extend(p.argmax(1).tolist())
            candidates.append((float(balanced_accuracy_score(yall,pall)),float(accuracy_score(yall,pall)),-c,c))
        choice=max(candidates);ablation[key]={"C":choice[3],"tuning_balanced_accuracy":choice[0],"tuning_accuracy":choice[1],"feature_count":int(len(ix))}
    return selections,ablation


def run(root=ROOT,output=OUT):
    os.environ["CUDA_VISIBLE_DEVICES"]="";os.environ["OMP_NUM_THREADS"]=str(THREADS)
    output=Path(output);output.mkdir(parents=True,exist_ok=True);ctx=common.load_context(root)
    X,names,owners=feature_matrix(ctx["texts"])
    start=time.monotonic();outer_preds=np.full(len(ctx["y"]),-1,dtype=np.int64);combined=np.full_like(outer_preds,-1);fold_rows=[];coefs=[];ablations={}
    with threadpool_limits(limits=THREADS):
        for outer in ctx["outer_folds"]:
            fold=outer["fold"];fit=outer["fit_rows"];held=outer["held_rows"]
            print(f"E17 outer fold {fold}/5 nested tuning",flush=True)
            params,ablation_params=_tune_outer(ctx,fit,X,names,owners)
            yfit=ctx["y"][fit]
            scaler,clf=_fit(X[fit],yfit,params["standalone_C"]);fp=_predict(scaler,clf,X[held]);standalone=fp.argmax(1)
            combined_cfg=params["combined"]
            cscaler,cclf=_fit(X[fit],yfit,combined_cfg["C"]);cp=_predict(cscaler,cclf,X[held])
            e13=outer["label_scores"];base=outer["baseline"];alt=outer["alternative"]
            joint,switch=_rerank(ctx["y"][held],base,alt,e13,cp,combined_cfg["weight"],combined_cfg["margin"])
            outer_preds[held]=standalone;combined[held]=joint
            foldab={}
            cat_names=sorted(set(owners.values()))
            for key,cfg in ablation_params.items():
                ix=np.asarray([i for i,n in enumerate(names) if (key=="all" or (key.startswith("without_") and owners[n]!=key.removeprefix("without_")) or (key.startswith("only_") and owners[n]==key.removeprefix("only_")))])
                ascaler,aclf=_fit(X[fit][:,ix],yfit,cfg["C"]);ap=_predict(ascaler,aclf,X[held][:,ix]);foldab[key]=ap.argmax(1)
                if key=="all": pass
            for key,p in foldab.items(): ablations.setdefault(key,np.full(len(ctx["y"]),-1,dtype=np.int64))[held]=p
            fold_info={"fold":fold,"rows":int(len(held)),"standalone_C":params["standalone_C"],
                "combined":combined_cfg,"standalone":common.metrics(ctx["y"][held],standalone),
                "combined_metrics":{**common.comparison(ctx["y"][held],base,joint),"correction_switches":int(switch.sum())},
                "e13_baseline":common.metrics(ctx["y"][held],base),
                "ablation_params":ablation_params,
                "per_label_gain_vs_E13":{lab:int(((joint==i)&(ctx["y"][held]==i)).sum()-((base==i)&(ctx["y"][held]==i)).sum()) for i,lab in enumerate(LABELS)}}
            fold_rows.append(fold_info);coefs.append(np.mean(np.abs(clf.coef_),axis=0))
            joblib.dump({"scaler":scaler,"classifier":clf,"feature_names":names,"feature_groups":owners,"C":params["standalone_C"]},output/f"E17_fold{fold}.joblib",compress=3)
            print(f"E17 fold {fold}: standalone BA={fold_info['standalone']['balanced_accuracy']:.4%}; combined BA={fold_info['combined_metrics']['balanced_accuracy']:.4%}",flush=True)
    if (outer_preds<0).any() or (combined<0).any():raise ValueError("E17 outer folds failed to cover V0")
    base=ctx["baseline"]; y=ctx["y"]
    stand=common.metrics(y,outer_preds); comb={**common.comparison(y,base,combined),"per_label":common.metrics(y,combined)["per_label"],"confusion_matrix":common.metrics(y,combined)["confusion_matrix"]}
    ablation_results={}
    for key,p in ablations.items():
        ablation_results[key]={**common.metrics(y,p),"delta_balanced_accuracy_vs_all":float(common.metrics(y,p)["balanced_accuracy"]-stand["balanced_accuracy"]),"feature_count":int(len([n for n in names if key=="all" or (key.startswith("without_") and owners[n]!=key.removeprefix("without_")) or (key.startswith("only_") and owners[n]==key.removeprefix("only_"))]))}
    coef=np.mean(coefs,axis=0); feature_importance=sorted([{"feature":n,"category":owners[n],"mean_abs_coefficient":float(v)} for n,v in zip(names,coef)],key=lambda x:-x["mean_abs_coefficient"])
    predictions=ctx["v0"][["id","prompt_id","model"]].rename(columns={"model":"y_true"}).copy();predictions["E13_B"]=np.asarray(LABELS,dtype=object)[base];predictions["E17_standalone"]=np.asarray(LABELS,dtype=object)[outer_preds];predictions["E17_combined"]=np.asarray(LABELS,dtype=object)[combined];predictions["combined_changed"]=combined!=base
    predictions["fold"]=-1
    for outer in ctx["outer_folds"]: predictions.loc[outer["held_rows"],"fold"]=outer["fold"]
    predictions.to_parquet(output/"predictions.parquet",index=False);np.savez_compressed(output/"predictions.npz",y=y,baseline=base,standalone=outer_preds,combined=combined,row_ids=ctx["row_ids"])
    result={"experiment":"E17","baseline":"E13-B Trust List 2 v037","validation_rows":len(y),"baseline_reproduced_exactly":True,
        "standalone":stand,"combined":comb,"folds":fold_rows,"feature_ablation":ablation_results,
        "feature_importance":feature_importance,"runtime_seconds":time.monotonic()-start,"threads":THREADS,
        "categories":sorted(set(owners.values())),"feature_count":len(names)}
    config={"seed":SEED,"threads":THREADS,"C_values":CS,"evidence_weights":WEIGHTS,"switch_margins":MARGINS,
        "label_order":LABELS,"feature_names":names,"feature_groups":owners,"raw_unicode_preserved":True,
        "baseline_run":str(common.e15.E13_RUN),"trust_list_snapshot":ctx["snapshot"],"cache_hashes":ctx["cache_hashes"]}
    (output/"config.json").write_text(json.dumps(config,indent=2,allow_nan=False)+"\n");(output/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");_write_report(output,result,feature_importance)
    print(f"E17 complete: standalone BA={stand['balanced_accuracy']:.4%}; combined BA={comb['balanced_accuracy']:.4%}",flush=True);return result


def _write_report(output,result,importance):
    s=result["standalone"];c=result["combined"]
    lines=["# E17 — Model Typography Fingerprints","",
      "This experiment uses explicit, Unicode-preserving formatting and typography counts from V0 raw responses. E13-B is reproduced from the saved five-fold models. All feature scaling, logistic fitting, C selection, evidence weighting, and switching thresholds use prompt-group training folds only. V0 remains a development evaluation because the E13 checkpoint inventory was selected against V0.","",
      "## Results","","| Method | Accuracy | Balanced accuracy | Δ BA vs E13-B | Corrected | Introduced | Net gains |","|---|---:|---:|---:|---:|---:|---:|",
      f"| E13-B | {c['baseline_accuracy']:.4%} | {c['baseline_balanced_accuracy']:.4%} | — | — | — | — |",
      f"| E17 standalone | {s['accuracy']:.4%} | {s['balanced_accuracy']:.4%} | {s['balanced_accuracy']-c['baseline_balanced_accuracy']:+.4%} | — | — | — |",
      f"| E13-B + fingerprint rerank | {c['accuracy']:.4%} | {c['balanced_accuracy']:.4%} | {c['delta_balanced_accuracy']:+.4%} | {c['corrected']} | {c['introduced']} | {c['net_gains']} |","",
      "E13-B reproduces exactly: **True**. The combined method considers the E13-B selected label and strongest reliability-supported alternative; it switches only when weighted fingerprint evidence overcomes the E13 score margin and learned threshold.","",
      "## Five outer folds","","| Fold | Rows | Standalone BA | Combined BA | Δ BA vs E13-B | Combined corrected | Introduced | C | Evidence weight | Switch margin |","|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for f in result["folds"]:
        q=f["combined_metrics"];cfg=f["combined"];lines.append(f"| {f['fold']} | {f['rows']} | {f['standalone']['balanced_accuracy']:.4%} | {q['balanced_accuracy']:.4%} | {q['delta_balanced_accuracy']:+.4%} | {q['corrected']} | {q['introduced']} | {cfg['C']:g} | {cfg['weight']:g} | {cfg['margin']:g} |")
    lines += ["","## Per-label results: combined method","","| Label | Support | Precision | Recall | F1 | TP | FP | FN | Δ correct vs E13-B |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for i,label in enumerate(LABELS):
        m=c["per_label"][label];gain=sum(f["per_label_gain_vs_E13"][label] for f in result["folds"])
        lines.append(f"| {label} | {m['support']} | {m['precision']:.4%} | {m['recall']:.4%} | {m['f1']:.4%} | {m['tp']} | {m['fp']} | {m['fn']} | {gain:+d} |")
    lines += ["","## Feature-category ablation","","Each `only_*` row fits a standalone classifier using that category. Each `without_*` row removes that category from the full fingerprint vector. The full model's C selection is performed within each outer training partition; ablation C values are also chosen by inner prompt-group validation.","","| Feature set | Features | Accuracy | Balanced accuracy | Δ BA vs full standalone |","|---|---:|---:|---:|---:|"]
    for key,m in sorted(result["feature_ablation"].items()):lines.append(f"| {key} | {m['feature_count'] if 'feature_count' in m else '—'} | {m['accuracy']:.4%} | {m['balanced_accuracy']:.4%} | {m['delta_balanced_accuracy_vs_all']:+.4%} |")
    lines += ["","## Most influential features","","Mean absolute logistic coefficients across outer-fold models (standardized training-fold feature space).","","| Feature | Category | Mean absolute coefficient |","|---|---|---:|"]
    for q in importance[:30]:lines.append(f"| `{q['feature']}` | {q['category']} | {q['mean_abs_coefficient']:.6g} |")
    lines += ["","## Artifacts","",f"- Analysis folder: `{output}`","- `predictions.parquet` and `predictions.npz`: row-aligned OOF predictions.","- `E17_fold*.joblib`: full-feature classifier and fold-local scaler.","- `config.json`, `results.json`: feature schema, fold choices and metrics.","- No model was submitted; E13-B remains unchanged.",""]
    text="\n".join(lines);(output/"e17.md").write_text(text);(ROOT/"docs/e17.md").write_text(text)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,default=ROOT);p.add_argument("--output",type=Path,default=OUT);a=p.parse_args(argv);run(a.root,a.output);return 0


if __name__=="__main__":raise SystemExit(main())
