"""Probability ensembles and leakage-safe cross-fitted global weights."""
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score
from .evaluation import ensemble_probabilities


def fit_weights(probabilities, y_true, classes, *, l2=.05):
    """Fit nonnegative weights by log loss plus shrinkage toward equal weights."""
    arrays=[np.asarray(p,dtype=float) for p in probabilities]
    if not arrays or any(a.shape != arrays[0].shape for a in arrays): raise ValueError("probability matrices must align")
    y=np.array([list(classes).index(v) for v in y_true]); target=np.eye(len(classes))[y]
    p=np.stack(arrays,axis=1)
    def objective(w):
        blend=np.einsum('m,nmc->nc',w,p)
        loss=-np.log(np.clip(blend[np.arange(len(y)),y],1e-12,1)).mean()
        return loss+l2*np.square(w-1/len(arrays)).sum()
    result=minimize(objective,np.full(len(arrays),1/len(arrays)),bounds=[(0,1)]*len(arrays),constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],method="SLSQP")
    if not result.success: raise RuntimeError(f"weight fit failed: {result.message}")
    return result.x


def cross_fitted_weights(probabilities, y_true, meta_folds, classes, *, l2=.05):
    """Each row is scored by weights fit on other prompt-group folds."""
    folds=np.asarray(meta_folds); outputs=np.zeros_like(np.asarray(probabilities[0]),dtype=float); weights={}
    for fold in np.unique(folds):
        train=folds!=fold; val=~train
        w=fit_weights([np.asarray(p)[train] for p in probabilities],np.asarray(y_true)[train],classes,l2=l2)
        outputs[val]=np.einsum('m,nmc->nc',w,np.stack([np.asarray(p)[val] for p in probabilities],axis=1)); weights[int(fold)]=w.tolist()
    return outputs,weights


def compare_ensembles(y_true, probability_by_model):
    names=list(probability_by_model); result={}
    for i in range(1,len(names)+1):
        selected=names[:i]; p=ensemble_probabilities([probability_by_model[n] for n in selected])
        result["+".join(selected)]={"members":selected,"probabilities":p,"accuracy":float(balanced_accuracy_score(y_true,np.asarray(sorted(set(y_true)))[p.argmax(1)]))}
    return result
