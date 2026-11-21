#!/usr/bin/env python3
"""Create source-ID conflict and V2.14 comparison tables for the v16 audit."""
from pathlib import Path
import pandas as pd
from src.data_factory import _normal_text
root=Path(__file__).resolve().parents[3]
out=root/'artifacts/analysis/openpool16_prompt_group_audit'
raw=pd.read_parquet(root/'data/raw/open_pool/v16/ntl_open_pool.parquet')
raw['prompt_canon']=raw.prompt_canon.fillna('').astype(str).str.strip()
raw['prompt_group']=raw.prompt_group.fillna('').astype(str).str.strip()
raw['source_dataset']=raw.source_dataset.fillna('').astype(str).str.strip()
raw['source_file']=raw.source_file.fillna('').astype(str).str.strip()
hascanon=raw.prompt_canon.ne('')
fallback_ids=set(raw.loc[~hascanon,'prompt_group'])
canonical_ids=set(raw.loc[hascanon,'prompt_group'])
fallback_canonical_overlap=fallback_ids & canonical_ids
fallback_namespace=raw.loc[~hascanon].groupby('prompt_group').agg(
    rows=('id','size'),source_datasets=('source_dataset','nunique'),source_files=('source_file','nunique')).reset_index()
namespace_summary={
    'fallback_source_prompt_ids':int(len(fallback_ids)),
    'fallback_ids_also_used_by_rows_with_prompt_canon':int(len(fallback_canonical_overlap)),
    'fallback_rows_using_an_id_also_used_with_prompt_canon':int(raw.loc[~hascanon,'prompt_group'].isin(fallback_canonical_overlap).sum()),
    'fallback_ids_shared_across_source_datasets':int(fallback_namespace.source_datasets.gt(1).sum()),
    'fallback_ids_reused_across_source_files':int(fallback_namespace.source_files.gt(1).sum()),
    'namespace_key':'source_dataset + source_file + prompt_group'}
(out/'fallback_prompt_id_namespace_audit.json').write_text(pd.Series(namespace_summary).to_json(indent=2))
prompts=pd.read_parquet(root/'data/raw/open_pool/v16/prompts.parquet')
prompts['_canon']=prompts.prompt_canon.fillna('').astype(str).map(_normal_text)
prompts['_text']=prompts.prompt_text.fillna('').astype(str)
ptext=prompts.groupby('_canon')._text.first().to_dict()
amb=[]
for (source,group),frame in raw.groupby(['source_dataset','prompt_group'],sort=True):
    keys=sorted({_normal_text(x) for x in frame.prompt_canon if str(x).strip()})
    if len(keys)>1:
        amb.append({'source_dataset':source,'prompt_group':group,'rows':len(frame),
          'source_files':'|'.join(sorted(frame.source_file.unique())),
          'distinct_prompt_canon_count':len(keys),'prompt_canon_sample':'|'.join(keys[:20]),
          'canonical_prompt_text_sample':' || '.join(dict.fromkeys(ptext.get(k,'') for k in keys[:5] if ptext.get(k,''))),
          'response_ids_sample':'|'.join(frame.id.astype(str).head(12)),
          'model_labels_sample':'|'.join(sorted(frame.model.astype(str).unique())[:20])})
pd.DataFrame(amb).sort_values('rows',ascending=False).to_csv(out/'source_prompt_id_conflicts.csv',index=False)
# V2.14 strict prompt keying: canonical prompt_canon, otherwise source_dataset + source prompt_id.
v=pd.read_parquet(root/'data/processed/dataset_v2.14.parquet')
v['prompt_canon']=v.prompt_canon.fillna('').astype(str).str.strip()
v['prompt_id']=v.prompt_id.fillna('').astype(str).str.strip()
v['source_dataset']=v.source_dataset.fillna('').astype(str).str.strip()
canon=v.prompt_canon.ne('')
has_b2m=v.source.notna() & v.source.astype(str).str.strip().ne('')
keyed=v[canon|has_b2m].copy()
isc=keyed.prompt_canon.ne('')
keyed['_key']=''
keyed.loc[isc,'_key']='canon:'+keyed.loc[isc,'prompt_canon'].map(_normal_text)
fb=~isc
keyed.loc[fb,'_key']='source:'+keyed.loc[fb,'source_dataset']+'\x1f'+keyed.loc[fb,'prompt_id'].map(_normal_text)
g=keyed.groupby('_key').agg(rows=('id','size'),labels=('model','nunique'))
gl=keyed.groupby(['_key','model']).size()
repeat_groups=int(gl.gt(1).groupby(level=0).any().sum())
rows=[]
for k in (2,3,4,5):
    e=g[g.labels.ge(k)]
    rows.append({'dataset':'V2.14','metric':f'min_{k}_distinct_labels_groups','value':int(len(e))})
    rows.append({'dataset':'V2.14','metric':f'min_{k}_distinct_labels_responses','value':int(e.rows.sum())})
    rows.append({'dataset':'V2.14','metric':f'min_{k}_distinct_labels_pairwise_comparisons_per_epoch','value':int((e.labels*(e.labels-1)//2).sum())})
v214={
 'classification_responses':len(v),'rows_with_reliable_prompt_key':len(keyed),'groups':len(g),
 'multi_label_groups':int(g.labels.ge(2).sum()),'multi_label_responses':int(g.loc[g.labels.ge(2),'rows'].sum()),
 'multi_label_response_pct_of_all':float(100*g.loc[g.labels.ge(2),'rows'].sum()/len(v)),
 'ranking_items_per_epoch':int(g.loc[g.labels.ge(2),'labels'].sum()),
 'pairwise_comparisons_per_epoch':int((g.loc[g.labels.ge(2),'labels']*(g.loc[g.labels.ge(2),'labels']-1)//2).sum()),
 'groups_with_repeated_same_model_responses':repeat_groups,
 'distinct_labels_coverage':int(keyed.loc[keyed._key.isin(g[g.labels.ge(2)].index),'model'].nunique()),
 'unkeyed_rows':int(len(v)-len(keyed)), 'source_namespaced_B2M_key_count':int(keyed.loc[fb,'_key'].nunique()),
 'source_id_values_shared_across_datasets_as_reported':267}
for k,val in v214.items(): rows.append({'dataset':'V2.14 strict source namespacing','metric':k,'value':val})
cur=pd.read_json(out/'summary.json',typ='series')
cur=cur.to_dict()
v16={
 'classification_responses':cur['valid_classification_responses'],'rows_with_reliable_prompt_key':cur['reliable_grouped_responses'],
 'groups':cur['reliable_prompt_groups'],'multi_label_groups':cur['groups_with_2plus_labels'],
 'multi_label_responses':cur['responses_in_groups_with_2plus_labels'],'multi_label_response_pct_of_all':cur['ranking_responses_pct_all'],
 'ranking_items_per_epoch':cur['ranking_items_per_epoch_min2'],'pairwise_comparisons_per_epoch':cur['pairwise_label_comparisons_per_epoch_min2'],
 'groups_with_repeated_same_model_responses':cur['groups_with_repeated_same_model'],'distinct_labels_coverage':cur['eligible_label_coverage'],
 'unkeyed_rows':cur['rows_missing_any_usable_id'],'source_namespaced_B2M_key_count':1811}
for metric,value in v16.items(): rows.append({'dataset':'Open Pool v16','metric':metric,'value':value})
for k in (2,3,4,5):
    f=cur['ranking_thresholds'][k-2]
    for metric in ('eligible_groups','eligible_responses','pairwise_label_comparisons_per_epoch'):
      rows.append({'dataset':'Open Pool v16','metric':f'min_{k}_distinct_labels_{metric}','value':f[metric]})
for metric,value in v214.items():
    if metric=='source_id_values_shared_across_datasets_as_reported': continue
    if metric in v16:
      rows.append({'dataset':'Difference (v16 - V2.14 strict)','metric':metric,'value':v16[metric]-value})
pd.DataFrame(rows).to_csv(out/'v2.14_comparison.csv',index=False)
(out/'v2.14_strict_metrics.json').write_text(pd.Series(v214).to_json(indent=2))
print('ambiguous prompt_group IDs',len(amb),'V2.14 strict comparison',v214)
