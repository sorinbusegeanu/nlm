"""Probability ensembles and leakage-safe cross-fitted global weights."""
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score
from .evaluation import ensemble_probabilities


def fit_weights(probabilities, y_true, classes, *, l2=.05):
    """Fit nonnegative weights by log loss plus shrinkage toward equal weights."""
    arrays=[np.asarray(p,dtype=float) for p in probabilities]
    if not arrays or any(a.shape != arrays[0].shape for a in arrays): raise ValueError("probability matrices must align")
    if arrays[0].ndim != 2 or arrays[0].shape[0] != len(y_true) or arrays[0].shape[1] != len(classes): raise ValueError("probabilities must have shape [rows, classes]")
    if any(not np.isfinite(a).all() or np.any(a < 0) or not np.allclose(a.sum(1),1,atol=1e-5) for a in arrays): raise ValueError("inputs must be valid probability matrices")
    classes=list(classes)
    if len(set(classes)) != len(classes): raise ValueError("classes must be unique")
    y=np.array([classes.index(v) for v in y_true])
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
    folds=np.asarray(meta_folds); y_true=np.asarray(y_true)
    if not probabilities or len(folds) != len(y_true) or any(len(p) != len(y_true) for p in probabilities): raise ValueError("all rows and meta folds must align")
    if folds.ndim != 1 or any(x is None for x in folds): raise ValueError("meta folds contain missing values")
    if np.issubdtype(folds.dtype, np.number) and np.isnan(folds.astype(float)).any(): raise ValueError("meta folds contain missing values")
    if len(np.unique(folds)) < 2: raise ValueError("cross-fitting requires at least two meta folds")
    outputs=np.zeros_like(np.asarray(probabilities[0]),dtype=float); weights={}
    for fold in np.unique(folds):
        train=folds!=fold; val=~train
        w=fit_weights([np.asarray(p)[train] for p in probabilities],np.asarray(y_true)[train],classes,l2=l2)
        outputs[val]=np.einsum('m,nmc->nc',w,np.stack([np.asarray(p)[val] for p in probabilities],axis=1)); weights[int(fold)]=w.tolist()
    return outputs,weights


def compare_ensembles(y_true, probability_by_model, classes=None):
    names=list(probability_by_model); result={}
    classes=list(classes) if classes is not None else sorted(set(y_true))
    for i in range(1,len(names)+1):
        selected=names[:i]; p=ensemble_probabilities([probability_by_model[n] for n in selected])
        result["+".join(selected)]={"members":selected,"probabilities":p,"accuracy":float(balanced_accuracy_score(y_true,np.asarray(classes)[p.argmax(1)]))}
    return result
