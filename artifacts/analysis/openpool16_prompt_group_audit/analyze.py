#!/usr/bin/env python3
"""Read-only audit of Open Pool v16 prompt groups for E21 eligibility."""
from __future__ import annotations
import hashlib, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/analysis/openpool16_prompt_group_audit'
RAW=ROOT/'data/raw/open_pool/v16/ntl_open_pool.parquet'
PROMPTS=ROOT/'data/raw/open_pool/v16/prompts.parquet'
V14=ROOT/'data/processed/dataset_v2.14.parquet'
V0=ROOT/'data/processed/dataset_v0.parquet'
CANONICAL={
'claude-fable-5','claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5',
'gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash',
'glm-5.3','glm-5.3-flash','gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra',
'gpt-6-astra','kimi-k3','qwen3.8-max'}
def norm(x):
    import unicodedata
    return ' '.join(unicodedata.normalize('NFKC',str(x)).casefold().split())
def clean(x): return '' if pd.isna(x) else str(x).strip()
def pct(a,b): return (100*a/b if b else 0.0)
def write(df,name): df.to_csv(OUT/name,index=False)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=pd.read_parquet(RAW).reset_index(drop=True)
    prompts=pd.read_parquet(PROMPTS).copy()
    v0=pd.read_parquet(V0)
    v14=pd.read_parquet(V14)
    n=len(d)
    assert n==231036
    for c in ['id','text','model','prompt_group','source_dataset','source_file','prompt_canon']:
        if c not in d: d[c]=''
        d[c]=d[c].fillna('').astype(str)
    d['model']=d.model.str.strip()
    d['_text_norm']=d.text.map(norm)
    d['_canon_norm']=d.prompt_canon.map(norm)
    d['_prompt_group']=d.prompt_group.str.strip()
    d['_source']=d.source_dataset.str.strip()
    d['_file']=d.source_file.str.strip()
    d['_valid_label']=d.model.isin(CANONICAL)
    # Canonical prompt table integrity and join coverage.
    prompts['canon_norm']=prompts.prompt_canon.fillna('').map(norm)
    prompts['prompt_text_norm']=prompts.prompt_text.fillna('').map(norm)
    pmap=prompts.groupby('canon_norm',dropna=False).agg(
        prompt_rows=('prompt_text','size'),distinct_texts=('prompt_text_norm','nunique'),
        canonical_text=('prompt_text','first'),panel_labels_min=('panel_labels','min'),
        panel_labels_max=('panel_labels','max'),passages_min=('passages','min'),passages_max=('passages','max')).reset_index()
    pmap=pmap[pmap.canon_norm.ne('')]
    pmap.to_csv(OUT/'canonical_prompt_metadata.csv',index=False)
    pkeys=set(pmap.canon_norm)
    pconf=set(pmap.loc[pmap.distinct_texts.gt(1),'canon_norm'])
    canonical_text_variants=pmap.set_index('canon_norm').distinct_texts.to_dict()
    # Source prompt IDs are evaluated as source/file-local; raw IDs are not assumed global.
    d['_source_key']=d['_source']+'\x1f'+d['_file']+'\x1f'+d['_prompt_group']
    d['_source_dataset_key']=d['_source']+'\x1f'+d['_prompt_group']
    d['_canonical_text_variants']=d['_canon_norm'].map(canonical_text_variants).fillna(0).astype(int)
    # A missing canon row only joins via a namespaced source ID. Any ID whose
    # supplied canonical mapping is conflicting is excluded from ranking.
    d['_group_key']=''
    hascanon=d._canon_norm.ne('')
    d.loc[hascanon,'_group_key']='canon:'+d.loc[hascanon,'_canon_norm']
    missing=~hascanon & d._source_key.ne('')
    d.loc[missing,'_group_key']='source:'+d.loc[missing,'_source_key']
    d['_key_basis']=np.where(hascanon,'prompt_canon',np.where(missing,'source_dataset+source_file+prompt_group','missing'))
    # Prompt ID mapping/namespace consistency.
    idmap=d[d._prompt_group.ne('')].groupby('_prompt_group').agg(
        rows=('id','size'),source_datasets=('_source','nunique'),source_files=('_file','nunique'),
        canon_keys=('_canon_norm',lambda s: s[s.ne('')].nunique()),labels=('model','nunique')).reset_index()
    idmap['requires_source_namespace']=idmap.source_datasets.gt(1)
    idmap['ambiguous_canonical_mapping']=idmap.canon_keys.gt(1)
    idmap['namespace_conflict_same_source_file']=(d[d._prompt_group.ne('')]
        .groupby('_prompt_group').apply(lambda g: bool(g.source_dataset.nunique()>1),include_groups=False).reindex(idmap._prompt_group).fillna(False).to_numpy()) if False else idmap.requires_source_namespace
    write(idmap,'source_prompt_id_mapping.csv')
    dsid=d[d._prompt_group.ne('')].groupby(['_source','_prompt_group'],dropna=False).agg(
        rows=('id','size'),source_files=('_file','nunique'),canon_keys=('_canon_norm',lambda s:s[s.ne('')].nunique()),
        distinct_canonical_texts=('_canonical_text_variants',lambda s:s[s.gt(0)].nunique()),
        labels=('model','nunique')).reset_index()
    dsid['file_local_collision']=dsid.source_files.gt(1)
    dsid['ambiguous_canonical_mapping']=dsid.canon_keys.gt(1)
    write(dsid,'source_namespaced_prompt_id_mapping.csv')
    # One-to-many canonical IDs and one-to-many prompt text/metadata.
    canon_rows=d[d._canon_norm.ne('')]
    canonmap=canon_rows.groupby('_canon_norm').agg(rows=('id','size'),source_datasets=('_source','nunique'),
        prompt_groups=('_prompt_group','nunique'),source_files=('_file','nunique'),labels=('model','nunique'),
        panel_labels_values=('panel_labels',lambda s: '|'.join(sorted(set(s.dropna().astype(str)))) if 'panel_labels' in d else ''),
        panel_labels_min=('panel_labels','min'),panel_labels_max=('panel_labels','max')).reset_index()
    canonmap['prompt_text_rows_present']=canonmap._canon_norm.isin(pkeys)
    canonmap['conflicting_prompt_text']=canonmap._canon_norm.isin(pconf)
    # Avoid aggregating absent publisher column.
    if 'panel_labels' in d:
        panel=d.groupby('_canon_norm').panel_labels.nunique(dropna=True)
        canonmap['panel_labels_inconsistent']=canonmap._canon_norm.map(panel).fillna(0).gt(1)
        canonmap['panel_labels_match_observed_distinct_labels']=canonmap.panel_labels_min.eq(canonmap.labels)
    else:
        canonmap['panel_labels_inconsistent']=False
        canonmap['panel_labels_match_observed_distinct_labels']=False
    write(canonmap,'canonical_group_metadata_consistency.csv')
    # Reliability: drop only ambiguous canonical text mappings, retain valid opaque canonical hashes.
    reliable=d._group_key.ne('') & (~d._canon_norm.isin(pconf)) & d._valid_label & d._text_norm.ne('')
    d['_reliable']=reliable
    dg=d.loc[reliable].copy()
    group=dg.groupby('_group_key',sort=True,dropna=False)
    gm=group.agg(rows=('id','size'),distinct_labels=('model','nunique'),source_datasets=('_source','nunique'),
        source_files=('_file','nunique'),key_basis=('_key_basis','first'),
        prompt_canon=('_canon_norm','first'),prompt_groups=('_prompt_group','nunique'),
        distinct_texts=('_text_norm','nunique'),id_count=('id','nunique')).reset_index()
    # Count duplicate model responses within each group and distinct output counts.
    gl=dg.groupby(['_group_key','model']).agg(label_rows=('id','size'),label_unique_texts=('_text_norm','nunique'),
        duplicate_text_excess=('_text_norm',lambda s:int(len(s)-s.nunique()))).reset_index()
    gm['labels_with_repeated_rows']=gm._group_key.map(gl[gl.label_rows.gt(1)].groupby('_group_key').size()).fillna(0).astype(int)
    gm['excess_same_label_response_rows']=gm._group_key.map(gl.groupby('_group_key').duplicate_text_excess.sum()).fillna(0).astype(int)
    gm['rows_within_repeated_label_groups']=gm._group_key.map(gl[gl.label_rows.gt(1)].groupby('_group_key').label_rows.sum()).fillna(0).astype(int)
    gm['group_repeated_label_outputs_all_distinct']=gm._group_key.map(
        gl[gl.label_rows.gt(1)].groupby('_group_key').apply(lambda g:bool((g.label_rows==g.label_unique_texts).all()),include_groups=False)).fillna(False)
    gm['contains_prompt_metadata_conflict']=gm._group_key.str.startswith('canon:') & gm.prompt_canon.isin(pconf)
    if 'panel_labels' in d:
        panel_by_key=d.groupby('_group_key').panel_labels.agg(panel_labels_min='min',panel_labels_max='max',panel_labels_values='nunique')
        gm=gm.merge(panel_by_key,on='_group_key',how='left',validate='one_to_one')
        gm['panel_labels_match_observed_distinct_labels']=gm.panel_labels_min.eq(gm.distinct_labels)
        gm['panel_labels_inconsistent_within_group']=gm.panel_labels_values.gt(1)
        gm[~gm.panel_labels_match_observed_distinct_labels].to_csv(OUT/'prompt_metadata_group_conflicts.csv',index=False)
    # all groups are canonical-label-only by construction; defensively record impossible over-18 groups.
    gm['more_than_18_labels']=gm.distinct_labels.gt(18)
    gm.to_csv(OUT/'reliable_group_inventory.csv',index=False)
    # response-count by group distinct-label-count.
    bylab=gm.groupby('distinct_labels').agg(groups=('_group_key','size'),responses=('rows','sum'),
        median_rows_per_group=('rows','median'),max_rows_per_group=('rows','max'),
        groups_with_repeated_model=('labels_with_repeated_rows',lambda s:int((s>0).sum())),
        repeated_label_rows=('rows_within_repeated_label_groups','sum'),same_label_duplicate_text_excess=('excess_same_label_response_rows','sum')).reset_index()
    bylab['group_share_pct']=bylab.groups / max(1,len(gm))*100
    bylab['response_share_pct']=bylab.responses / max(1,len(dg))*100
    write(bylab,'groups_by_distinct_label_count.csv')
    size=gm.groupby('rows').agg(groups=('_group_key','size'),responses=('rows','sum')).reset_index().sort_values('rows')
    size['group_share_pct']=size.groups/len(gm)*100
    size['response_share_pct']=size.responses/len(dg)*100
    write(size,'group_size_distribution_exact.csv')
    # Group-size bins across actual response rows, plus label diversity buckets.
    buckets=[]
    for name,lo,hi in [('1',1,1),('2',2,2),('3',3,3),('4-5',4,5),('6-9',6,9),('10-14',10,14),('15-18',15,18)]:
        sel=gm[gm.distinct_labels.between(lo,hi)]
        buckets.append({'distinct_label_bin':name,'groups':len(sel),'responses':int(sel.rows.sum()),
            'group_share_pct':pct(len(sel),len(gm)),'response_share_pct':pct(sel.rows.sum(),len(dg))})
    write(pd.DataFrame(buckets),'groups_by_label_bands.csv')
    # Training populations for group minimums 2..5; thresholds are distinct labels.
    thresholds=[]
    for k in (2,3,4,5):
        elig=gm[gm.distinct_labels.ge(k)]
        thresholds.append({'minimum_distinct_labels':k,'eligible_groups':len(elig),
            'eligible_responses':int(elig.rows.sum()),'excluded_groups':int(len(gm)-len(elig)),
            'excluded_responses':int(len(dg)-elig.rows.sum()),
            'eligible_response_pct_of_reliable_grouped':pct(elig.rows.sum(),len(dg)),
            'eligible_response_pct_of_all_v16':pct(elig.rows.sum(),n),
            'ranking_items_per_epoch':int(elig.distinct_labels.sum()),
            'pairwise_label_comparisons_per_epoch':int((elig.distinct_labels*(elig.distinct_labels-1)//2).sum()),
            'groups_with_repeated_same_model_rows':int((elig.labels_with_repeated_rows>0).sum()),
            'groups_with_same_label_duplicate_responses':int((elig.excess_same_label_response_rows>0).sum())})
    thr=pd.DataFrame(thresholds)
    write(thr,'ranking_minimum_label_thresholds.csv')
    # Classification eligibility is all valid labeled non-empty responses; not restricted by prompt quality.
    valid=d[d._valid_label & d._text_norm.ne('')]
    label_counts=valid.groupby('model').size().reindex(sorted(CANONICAL),fill_value=0).rename('classification_responses').rename_axis('label').reset_index()
    rankk=gm[gm.distinct_labels.ge(2)]
    rank_rows=dg[dg._group_key.isin(set(rankk._group_key))]
    rank_counts=rank_rows.groupby('model').size().reindex(sorted(CANONICAL),fill_value=0).rename('ranking_responses').rename_axis('label').reset_index()
    label=label_counts.merge(rank_counts,on='label',how='outer').fillna(0)
    label['ranking_response_pct_of_classification']=label.apply(lambda r:pct(r.ranking_responses,r.classification_responses),axis=1)
    write(label,'eligible_responses_by_label.csv')
    # Duplicate normalized response diagnostics.
    tx=d[d._text_norm.ne('')].groupby('_text_norm').agg(rows=('id','size'),ids=('id',lambda s:'|'.join(s.astype(str).head(8))),
        label_count=('model','nunique'),labels=('model',lambda s:'|'.join(sorted(set(s.astype(str))))),
        group_count=('_group_key','nunique'),sources=('_source',lambda s:'|'.join(sorted(set(s.astype(str)))[:5]))).reset_index()
    dup=tx[tx.rows.gt(1)].copy(); dup['conflicting_labels']=dup.label_count.gt(1)
    write(dup.sort_values(['conflicting_labels','rows'],ascending=[False,False]).head(5000),'duplicate_normalized_responses.csv')
    dup_summary={'normalized_text_duplicate_clusters':int(len(dup)),'duplicate_excess_rows':int((dup.rows-1).sum()),
       'rows_in_duplicate_clusters':int(dup.rows.sum()),'conflicting_label_duplicate_clusters':int(dup.conflicting_labels.sum()),
       'conflicting_label_excess_rows':int(dup.loc[dup.conflicting_labels,'rows'].sum()-dup.conflicting_labels.sum()),
       'same_label_duplicate_clusters':int((~dup.conflicting_labels).sum())}
    # Per group-label repetition analysis; unique outputs vs duplicates.
    repeated=gl[gl.label_rows.gt(1)].copy()
    repeated['all_outputs_distinct']=repeated.label_rows.eq(repeated.label_unique_texts)
    repeated.to_csv(OUT/'repeated_model_labels_by_group.csv',index=False)
    repeats_summary={'groups_with_repeated_model_label':int((gm.labels_with_repeated_rows>0).sum()),
        'group_label_pairs_with_repeated_rows':int(len(repeated)),
        'repeated_label_response_rows':int(repeated.label_rows.sum()),
        'repeated_label_group_pairs_all_distinct_outputs':int(repeated.all_outputs_distinct.sum()),
        'repeated_label_group_pairs_with_exact_duplicate_output':int((~repeated.all_outputs_distinct).sum()),
        'excess_exact_same_label_response_rows_within_groups':int(repeated.duplicate_text_excess.sum()),
        'repeated_label_outputs_distinct_pct':pct(repeated.all_outputs_distinct.sum(),len(repeated))}
    # Prompt metadata/key reliability and conflicts.
    groupby_source= idmap
    stats={
      'dataset_rows':n,'dataset_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),
      'prompt_metadata_rows':len(prompts),'prompt_metadata_sha256':hashlib.sha256(PROMPTS.read_bytes()).hexdigest(),
      'canonical_prompt_unique_keys':int(prompts.canon_norm.nunique()),'canonical_prompt_keys_joined':int(d._canon_norm[d._canon_norm.ne('')].nunique()),
      'rows_with_prompt_canon':int(hascanon.sum()),'rows_missing_prompt_canon':int((~hascanon).sum()),
      'prompt_canon_missing_pct':pct((~hascanon).sum(),n),'rows_with_prompt_group':int(d._prompt_group.ne('').sum()),
      'rows_missing_prompt_group':int(d._prompt_group.eq('').sum()),'rows_missing_any_usable_id':int((d._canon_norm.eq('') & d._prompt_group.eq('')).sum()),
      'prompt_canon_keys_with_no_metadata_row':int(len(set(d.loc[hascanon,'_canon_norm'])-pkeys)),
      'canonical_prompt_keys_with_text':int(len(pkeys)),
      'canonical_prompt_keys_without_text':int(len(set(d.loc[hascanon,'_canon_norm'])-pkeys)),
      'canonical_prompt_rows_without_prompt_text':int((hascanon & ~d._canon_norm.isin(pkeys)).sum()),
      'prompt_canon_keys_with_conflicting_canonical_text':int(len(pconf)),
      'rows_with_conflicting_canonical_prompt_text':int(d._canon_norm.isin(pconf).sum()),
      'unique_source_prompt_ids':int(d._prompt_group[d._prompt_group.ne('')].nunique()),
      'source_prompt_ids_shared_across_datasets':int(idmap.requires_source_namespace.sum()),
      'rows_under_shared_source_prompt_ids':int(idmap.loc[idmap.requires_source_namespace,'rows'].sum()),
      'source_prompt_ids_mapping_multiple_prompt_canon':int(idmap.ambiguous_canonical_mapping.sum()),
      'source_prompt_ids_multi_canon_after_dataset_namespace':int(dsid.ambiguous_canonical_mapping.sum()),
      'source_dataset_prompt_ids_reused_across_files':int(dsid.file_local_collision.sum()),
      'canonical_groups_spanning_sources':int((canonmap.source_datasets>1).sum()),
      'canonical_groups_spanning_multiple_prompt_group_ids':int((canonmap.prompt_groups>1).sum()),
      'canonical_groups_with_inconsistent_panel_labels':int(canonmap.panel_labels_inconsistent.sum()),
      'canonical_groups_where_panel_labels_equal_observed_distinct_labels':int(canonmap.panel_labels_match_observed_distinct_labels.sum()),
      'canonical_groups_where_panel_labels_differ_from_observed_distinct_labels':int((~canonmap.panel_labels_match_observed_distinct_labels).sum()),
      'all_groups_where_panel_labels_differ_from_observed_distinct_labels':int((~gm.panel_labels_match_observed_distinct_labels).sum()),
      'all_group_rows_with_panel_label_mismatch':int(gm.loc[~gm.panel_labels_match_observed_distinct_labels,'rows'].sum()),
      'fallback_groups_where_panel_labels_differ_from_observed_distinct_labels':int(((gm.key_basis!='prompt_canon') & ~gm.panel_labels_match_observed_distinct_labels).sum()),
      'fallback_group_rows_with_panel_label_mismatch':int(gm.loc[(gm.key_basis!='prompt_canon') & ~gm.panel_labels_match_observed_distinct_labels,'rows'].sum()),
      'canonical_prompt_metadata_groups':int(len(pmap)),
      'canonical_prompt_metadata_keys_with_conflicts':int(pmap.distinct_texts.gt(1).sum()),
      'canonical_prompt_text_missing_for_metadata_key':int((pmap.prompt_rows==0).sum()),
      'unreliable_canonical_conflict_rows_excluded_from_grouping':int(d._canon_norm.isin(pconf).sum()),
      'response_ids_unique':bool(d.id.is_unique),'duplicate_response_id_rows':int(d.id.duplicated(keep=False).sum()),
      'valid_classification_responses':int(len(valid)),
      'invalid_or_empty_classification_rows':int(n-len(valid)),
      'invalid_label_rows':int((~d._valid_label).sum()),'empty_text_rows':int(d._text_norm.eq('').sum()),
      'classification_label_coverage':int(valid.model.nunique()),
      'reliable_grouped_responses':int(len(dg)),'reliable_prompt_groups':int(len(gm)),
      'rows_excluded_from_reliable_grouping':int(n-len(dg)),
      'groups_with_1_label':int((gm.distinct_labels==1).sum()),
      'groups_with_2plus_labels':int((gm.distinct_labels>=2).sum()),
      'responses_in_groups_with_2plus_labels':int(gm.loc[gm.distinct_labels>=2,'rows'].sum()),
      'ranking_responses_pct_all':pct(gm.loc[gm.distinct_labels>=2,'rows'].sum(),n),
      'ranking_responses_pct_reliable_grouped':pct(gm.loc[gm.distinct_labels>=2,'rows'].sum(),len(dg)),
      'eligible_label_coverage':int(rank_rows.model.nunique()),
      'max_group_rows':int(gm.rows.max()),'median_group_rows':float(gm.rows.median()),
      'mean_group_rows':float(gm.rows.mean()),'p95_group_rows':float(gm.rows.quantile(.95)),
      'max_distinct_labels':int(gm.distinct_labels.max()),
      'groups_more_than_18_distinct_labels':int(gm.more_than_18_labels.sum()),
      'groups_with_repeated_same_model':int((gm.labels_with_repeated_rows>0).sum()),
      'groups_with_exact_same_model_response_duplicates':int((gm.excess_same_label_response_rows>0).sum()),
      'groups_with_all_distinct_outputs_for_repeated_labels':int(gm.group_repeated_label_outputs_all_distinct.sum()),
      'responses_in_repeated_label_groups':int(gm.rows_within_repeated_label_groups.sum()),
      'repeated_label_exact_duplicate_excess_rows':int(gm.excess_same_label_response_rows.sum()),
      'one_ranking_set_per_multilabel_group_per_epoch':int((gm.distinct_labels.ge(2)).sum()),
      'ranking_items_per_epoch_min2':int(gm.loc[gm.distinct_labels.ge(2),'distinct_labels'].sum()),
      'pairwise_label_comparisons_per_epoch_min2':int((gm.loc[gm.distinct_labels.ge(2),'distinct_labels']*(gm.loc[gm.distinct_labels.ge(2),'distinct_labels']-1)//2).sum()),
      'group_sizes_summary':{'min':int(gm.rows.min()),'median':float(gm.rows.median()),'mean':float(gm.rows.mean()),'p90':float(gm.rows.quantile(.9)),'p95':float(gm.rows.quantile(.95)),'p99':float(gm.rows.quantile(.99)),'max':int(gm.rows.max())},
      'duplicate_summary':dup_summary,'repeated_label_summary':repeats_summary}
    # V0 overlap exact IDs, normalized outputs, and prompt IDs (identifiers only).
    v0_ids=set(v0.id.fillna('').astype(str)); v0_norm=set(v0.text.fillna('').astype(str).map(norm))-{''}; v0_pids=set(v0.prompt_id.fillna('').astype(str).map(norm))-{''}
    open_ids=set(d.id); open_norm=set(d._text_norm)-{''}
    canon_keys=set(d._canon_norm)-{''}; source_keys=set(d._prompt_group.map(norm))-{''}
    stats['v0_overlap']={
      'response_id_overlap_rows':int(d.id.isin(v0_ids).sum()),'unique_response_ids_overlapping':int(len(set(d.id)&v0_ids)),
      'normalized_response_text_overlap_rows':int(d._text_norm.isin(v0_norm).sum()),'unique_normalized_text_overlap':int(len(open_norm&v0_norm)),
      'prompt_canon_values_matching_v0_prompt_ids':int(len(canon_keys&v0_pids)),
      'source_prompt_group_values_matching_v0_prompt_ids':int(len(source_keys&v0_pids)),
      'openpool_rows_with_prompt_canon_matching_v0_prompt_id':int(d._canon_norm.isin(v0_pids).sum()),
      'openpool_rows_with_prompt_group_matching_v0_prompt_id':int(d._prompt_group.map(norm).isin(v0_pids).sum()),
      'note':'V0 contains opaque prompt_id values but no original prompt text; exact identifier overlap is only a direct-key check.'}
    # Response-label consistency: canonical labels, lab family, and served model when canonicalizable.
    def family(label):
      if label.startswith('claude-'): return 'anthropic'
      if label.startswith('gemini-'): return 'google'
      if label.startswith('glm-'): return 'zai'
      if label.startswith('gpt-'): return 'openai'
      if label.startswith('kimi-'): return 'moonshot'
      if label.startswith('qwen'): return 'qwen'
      return 'unknown'
    if 'lab' in d:
      d['_expected_lab']=d.model.map(family)
      d['_lab']=d.lab.fillna('').astype(str).str.strip().str.casefold()
      lab_conf=d[d._valid_label & d._lab.ne('') & d._lab.ne(d._expected_lab)]
      stats['label_metadata']={'rows_with_lab':int(d._lab.ne('').sum()),'lab_family_conflicts':len(lab_conf),
        'conflict_rows_by_label':lab_conf.model.value_counts().to_dict(),
        'label_lab_cross_tab':pd.crosstab(d.model,d._lab).to_dict()}
      write(lab_conf[['id','model','lab','source_dataset','source_file','prompt_group','prompt_canon']].head(500), 'label_metadata_conflicts.csv')
    if 'gen_served_model' in d:
      canonical_by_key={''.join(ch for ch in label.casefold() if ch.isalnum()):label for label in CANONICAL}
      served=d.gen_served_model.fillna('').astype(str).str.strip().str.rsplit('/',n=1).str[-1]
      served_key=served.map(lambda value: ''.join(ch for ch in value.casefold() if ch.isalnum()))
      served_label=served_key.map(canonical_by_key)
      served_bad=d[d._valid_label & served_label.notna() & served_label.ne(d.model)]
      stats['label_metadata']['known_canonical_served_model_mismatches']=int(len(served_bad))
      write(served_bad[['id','model','gen_served_model','source_dataset','source_file','prompt_group','prompt_canon']].head(500), 'served_model_label_conflicts.csv')
    # IDs/texts problematic group sample: oversized groups, repeated outputs, metadata conflicts.
    issues=[]
    for _,g in gm.sort_values(['rows','distinct_labels'],ascending=False).head(50).iterrows():
      if g.rows>18: issues.append({'group_key':g._group_key,'issue':'more_than_18_response_rows_due_to_repeated_labels','rows':int(g.rows),'distinct_labels':int(g.distinct_labels),'source_datasets':int(g.source_datasets),'prompt_group_ids':int(g.prompt_groups),'example_ids':''})
    for _,r in repeated.sort_values(['duplicate_text_excess','label_rows'],ascending=False).head(100).iterrows():
      sub=dg[(dg._group_key==r._group_key)&(dg.model==r.model)]
      issues.append({'group_key':r._group_key,'issue':'repeated_model_label_exact_duplicate_output' if r.duplicate_text_excess else 'repeated_model_label_distinct_outputs',
       'rows':int(r.label_rows),'distinct_labels':int(gm.loc[gm._group_key==r._group_key,'distinct_labels'].iloc[0]),
       'source_datasets':'|'.join(sorted(sub._source.unique())[:5]),'prompt_group_ids':'|'.join(sorted(sub._prompt_group.unique())[:5]),
       'model':r.model,'response_ids':'|'.join(sub.id.head(8)),'example_texts':' || '.join(sub.text.head(3).str.slice(0,220))})
    for key in sorted(pconf):
      sub=d[d._canon_norm==key]
      issues.append({'group_key':'canon:'+key,'issue':'conflicting_canonical_prompt_text_metadata','rows':len(sub),'distinct_labels':sub.model.nunique(),
       'source_datasets':'|'.join(sorted(sub._source.unique())[:5]),'prompt_group_ids':'|'.join(sorted(sub._prompt_group.unique())[:5]),'response_ids':'|'.join(sub.id.head(8)),
       'example_texts':' || '.join(sub.text.head(3).str.slice(0,220))})
    pd.DataFrame(issues).drop_duplicates().head(300).to_csv(OUT/'problematic_group_examples.csv',index=False)
    # Distinct labels and prompt identifier distribution for rows/key bases.
    keybasis=d.groupby('_key_basis').agg(rows=('id','size'),valid_labeled_rows=('_valid_label','sum'),
       unique_prompt_ids=('_prompt_group','nunique'),unique_canon=('_canon_norm',lambda s:s[s.ne('')].nunique())).reset_index()
    keybasis.to_csv(OUT/'key_basis_coverage.csv',index=False)
    gb=[]
    for basis, frame in gm.groupby('key_basis',sort=True):
      eligible=frame[frame.distinct_labels.ge(2)]
      gb.append({'key_basis':basis,'groups':len(frame),'responses':int(frame.rows.sum()),
        'multi_label_groups':len(eligible),'multi_label_responses':int(eligible.rows.sum()),
        'ranking_items_per_epoch':int(eligible.distinct_labels.sum()),
        'pairwise_label_comparisons_per_epoch':int((eligible.distinct_labels*(eligible.distinct_labels-1)//2).sum()),
        'groups_with_repeated_model_rows':int(frame.labels_with_repeated_rows.gt(0).sum()),
        'groups_with_exact_same_model_response_duplicates':int(frame.excess_same_label_response_rows.gt(0).sum())})
    pd.DataFrame(gb).to_csv(OUT/'group_eligibility_by_key_basis.csv',index=False)
    label_dist=d.groupby('model').size().reindex(sorted(CANONICAL),fill_value=0).rename('responses').rename_axis('label').reset_index()
    label_dist['share_pct']=label_dist.responses/n*100
    write(label_dist,'classification_label_distribution.csv')
    stats['row_counts_by_key_basis']=keybasis.to_dict('records')
    stats['label_counts']=dict(zip(label_dist.label,label_dist.responses.astype(int)))
    stats['rank_label_counts']=dict(zip(label.label,label.ranking_responses.astype(int)))
    stats['ranking_thresholds']=thr.to_dict('records')
    stats['analysis_definition']={
      'classification':'Every row with a canonical 18-class model label and nonempty response text, regardless of prompt metadata quality.',
      'grouping_key':'NFKC/casefold/whitespace-normalized nonempty prompt_canon; when absent, source_dataset + source_file + prompt_group. Canonical prompt keys whose prompt table contains conflicting normalized prompt text are excluded from ranking.',
      'ranking_eligibility':'Reliable group with at least two distinct canonical model labels. At each ranking step, one response per label; duplicate-label outputs rotate across epochs.',
      'comparisons':'One ranking step per eligible group per epoch; pairwise label comparisons sum choose(distinct_label_count,2). Duplicate responses do not multiply comparisons within an epoch.'}
    (OUT/'summary.json').write_text(json.dumps(stats,indent=2,default=str))
    print(json.dumps(stats,indent=2,default=str))
if __name__=='__main__': main()
