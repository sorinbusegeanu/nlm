import numpy as np, pandas as pd
FEATURES=("checkpoint_id","model_family","predicted_label","competing_label","top1_probability","top2_probability","margin","normalized_entropy","label_precision","label_recall","pred_to_competitor_rate","competitor_to_pred_rate","agreement_count","family_agreement_count","ensemble_pred_probability","ensemble_competitor_probability","checkpoint_ensemble_pred_delta","checkpoint_ensemble_competitor_delta")
CATS=("checkpoint_id","model_family","predicted_label","competing_label")
def make_features(probs,members,classes,stats):
    p=np.asarray(probs,dtype=np.float64); ids=[m["candidate_id"] for m in members]; fam=[m["family"] for m in members]; m,n,k=p.shape
    assert p.shape[0]==len(ids)==len(fam) and k==len(classes) and np.isfinite(p).all() and np.allclose(p.sum(2),1,atol=1e-4)
    top=p.argmax(2); ordered=np.sort(p,axis=2); top1=ordered[:,:,-1]; top2=ordered[:,:,-2]; margin=top1-top2
    entropy=-(np.clip(p,1e-12,1)*np.log(np.clip(p,1e-12,1))).sum(2)/np.log(k)
    names=sorted(set(fam)); family_probs=np.mean([p[np.asarray(fam)==f].mean(0) for f in names],axis=0); family_probs/=family_probs.sum(1,keepdims=True)
    rank=np.argsort(-family_probs,axis=1,kind="stable"); ep=family_probs.argmax(1)
    competing=np.where(top==ep[None,:],rank[:,1][None,:],ep[None,:])
    for j in range(m):
        same=top[j]==ep
        competing[j,same]=rank[same,1]
    precision=np.asarray(stats["precision"],float); recall=np.asarray(stats["recall"],float); directional=np.asarray(stats["directional"],float)
    row=np.arange(n); ownp=family_probs[row[None,:],top]; altp=family_probs[row[None,:],competing]
    owncp=p[np.arange(m)[:,None],row[None,:],top]; altcp=p[np.arange(m)[:,None],row[None,:],competing]
    agreements=(top[:,None,:]==top[None,:,:]).sum(1)-1; famagree=np.zeros((m,n),float)
    for j in range(m):
        famagree[j]=sum(np.any(top[np.asarray(fam)==f]==top[j],axis=0) for f in names if f!=fam[j])
    frame=pd.DataFrame({"checkpoint_id":np.tile(np.asarray(ids,dtype=str),n),"model_family":np.tile(np.asarray(fam,dtype=str),n),"predicted_label":np.asarray(classes,dtype=object)[top].T.reshape(-1),"competing_label":np.asarray(classes,dtype=object)[competing].T.reshape(-1),"top1_probability":top1.T.reshape(-1),"top2_probability":top2.T.reshape(-1),"margin":margin.T.reshape(-1),"normalized_entropy":entropy.T.reshape(-1),"label_precision":precision[np.arange(m)[:,None],top].T.reshape(-1),"label_recall":recall[np.arange(m)[:,None],top].T.reshape(-1),"pred_to_competitor_rate":directional[np.arange(m)[:,None],top,competing].T.reshape(-1),"competitor_to_pred_rate":directional[np.arange(m)[:,None],competing,top].T.reshape(-1),"agreement_count":agreements.T.reshape(-1),"family_agreement_count":famagree.T.reshape(-1),"ensemble_pred_probability":ownp.T.reshape(-1),"ensemble_competitor_probability":altp.T.reshape(-1),"checkpoint_ensemble_pred_delta":(owncp-ownp).T.reshape(-1),"checkpoint_ensemble_competitor_delta":(altcp-altp).T.reshape(-1)})
    for c in CATS: frame[c]=frame[c].astype(str)
    return frame.loc[:,FEATURES]
def family_max_scores(hard,reliability,probs,families):
    hard=np.asarray(hard,dtype=np.int64); reliability=np.asarray(reliability,dtype=np.float64); probs=np.asarray(probs,dtype=np.float64)
    if hard.ndim!=2 or reliability.shape!=hard.shape or probs.ndim!=2 or probs.shape[0]!=hard.shape[1]:
        raise ValueError("family-max evidence arrays are misaligned")
    m,n=hard.shape; k=probs.shape[1]; scores=np.full((n,k),-np.inf)
    for r in range(n):
        for label in range(k):
            voters=np.flatnonzero(hard[:,r]==label)
            if len(voters):
                by={}
                for i in voters: by[families[i]]=max(by.get(families[i],-np.inf),reliability[i,r])
                scores[r,label]=np.mean(list(by.values()))
    safe=scores.copy()
    for r in range(n):
        supported=np.isfinite(scores[r])
        if not supported.any(): raise ValueError(f"no checkpoint supports row {r}")
        floor=float(scores[r,supported].min())-1.0
        tie=probs[r]/max(float(probs[r].max()),1e-12)
        safe[r,supported]+=tie[supported]*1e-12
        safe[r,~supported]=floor+tie[~supported]*1e-9
    return scores,safe

def route_family_max(hard,reliability,probs,families):
    scores,_=family_max_scores(hard,reliability,probs,families)
    pred=np.empty(len(scores),dtype=np.int64)
    for row,s in enumerate(scores):
        best=np.max(s[np.isfinite(s)]); tied=np.flatnonzero(np.isclose(s,best,rtol=0,atol=1e-12))
        pred[row]=min(tied,key=lambda l:(-probs[row,l],l))
    return pred
