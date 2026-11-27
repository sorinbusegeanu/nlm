"""Build two Kaggle notebooks that blend E41 with E13-B and E33-F."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import nbformat

from scripts.kaggle_preflight import embed_runtime_preflight, validate_kernel_package

ROOT = Path(__file__).resolve().parents[1]
AUX = ROOT / "artifacts/kaggle/e42-e43-feature-members-v001"
AUX_REF = "zodrak/e38-e39-feature-members-v001"
CACHE = ROOT / "artifacts/kaggle/e42-e43-tiktoken-cache-v001"
CACHE_REF = "zodrak/e42-e43-tiktoken-cache-v001"
PULLS = {
    "e13": ("notebooks/kaggle_e13b_v037_source.ipynb",
            "notebooks/kaggle_e13b_v037_source_metadata.json"),
    "e33": ("notebooks/kaggle_e33f_e31_40_source.ipynb",
            "notebooks/kaggle_e33f_e31_40_source_metadata.json"),
}
OUTPUTS = {
    "E42": ("artifacts/kaggle/e42-kernel", "zodrak/e42-e13-b-plus-e41",
            "E42_E13B_plus_E41.ipynb"),
    "E43": ("artifacts/kaggle/e43-kernel", "zodrak/e43-e33-f-plus-e41",
            "E43_E33F_plus_E41.ipynb"),
}
FAMILY_MAX_SCORES_COMPAT = '''\ndef family_max_scores(hard,reliability,probs,families):
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
'''

E41_BOOTSTRAP = r'''
# Load, compatibility-patch and smoke-test E38/E39 before checkpoint inference.
AUXROOT=root_for('e38_e39_manifest.json')
AUXMAN=json.loads((AUXROOT/'e38_e39_manifest.json').read_text())
assert AUXMAN['dataset_id']=='zodrak/e38-e39-feature-members-v001'
assert AUXMAN['classes']==B['classes'], 'E38/E39 label ordering differs from deployment'
for asset in AUXMAN['assets']:
    p=AUXROOT/Path(asset['flat_filename']).name
    assert p.is_file() and p.stat().st_size==int(asset['size']), f"missing/size mismatch: {p}"
    digest=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    h=digest.hexdigest()
    assert h==asset['sha256'], f"asset hash mismatch: {p}"
TIKROOT=root_for('tiktoken_cache_manifest.json')
TKMAN=json.loads((TIKROOT/'tiktoken_cache_manifest.json').read_text())
assert TKMAN['dataset_id']=='zodrak/e42-e43-tiktoken-cache-v001'
tik_cache=Path('/kaggle/working/tiktoken_cache'); tik_cache.mkdir(parents=True,exist_ok=True)
for asset in TKMAN['assets']:
    p=TIKROOT/asset['flat_filename']
    assert p.is_file() and p.stat().st_size==int(asset['size']), f"tiktoken cache missing/size mismatch: {p}"
    digest=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    assert digest.hexdigest()==asset['sha256'], f"tiktoken cache hash mismatch: {p}"
    shutil.copy2(p,tik_cache/asset['flat_filename'])
os.environ['TIKTOKEN_CACHE_DIR']=str(tik_cache)
import importlib.util, src
feature_path=AUXROOT/'runtime_feature_factory.py'
spec=importlib.util.spec_from_file_location('src.feature_factory',feature_path)
assert spec is not None and spec.loader is not None
feature_factory=importlib.util.module_from_spec(spec)
sys.modules['src.feature_factory']=feature_factory
spec.loader.exec_module(feature_factory)
sys.path.insert(0,str(AUXROOT))
from e38_e39_inference import predict_e38, predict_e39, blend_equal
smoke_text=['Offline E38 E39 Kaggle pre-inference compatibility smoke test.']
smoke38=predict_e38(AUXROOT/'e38_svc_bundle.joblib',smoke_text,B['classes'])
e39_dir=Path('/kaggle/working/e39_feature_model'); tokdir=e39_dir/'tokenizers'
e39_dir.mkdir(parents=True,exist_ok=True)
shutil.copy2(AUXROOT/'e39_model.cbm',e39_dir/'model.cbm')
shutil.copy2(AUXROOT/'e39_stylometry.json',e39_dir/'stylometry.json')
tokenizer_map=AUXMAN['components']['E39']['tokenizer_files']
for tok_name,files in tokenizer_map.items():
    dest=tokdir/tok_name; dest.mkdir(parents=True,exist_ok=True)
    for original,packaged in files.items(): shutil.copy2(AUXROOT/packaged,dest/original)
smoke39=predict_e39(e39_dir,smoke_text,B['classes'],feature_factory)
assert smoke38.shape==smoke39.shape==(1,len(B['classes']))
assert np.isfinite(smoke38).all() and np.isfinite(smoke39).all()
'''

E41_INFERENCE = r'''
# E38 + E39 probabilities on competition rows; same feature pipeline as training.
n=len(texts)
feature_started=time.time()
p38=predict_e38(AUXROOT/'e38_svc_bundle.joblib',texts,classes)
p39=predict_e39(e39_dir,texts,classes,feature_factory)
assert p38.shape==p39.shape==(n,len(classes))
assert np.isfinite(p38).all() and np.isfinite(p39).all()
e41_probs=0.5*p38+0.5*p39
assert np.allclose(e41_probs.sum(1),1,atol=1e-8)
e38_e39_seconds=time.time()-feature_started
del p38,p39; gc.collect()
'''


def _append_source(dataset_ref: str, title: str, output_dir: Path,
                   kernel_ref: str, code_name: str, base_nb: Path,
                   base_meta: Path, experiment: str) -> dict:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    shutil.copy2(AUX / "e38_e39_manifest.json", output_dir / "e38_e39_manifest.json")
    nb = nbformat.read(base_nb, as_version=4)
    metadata = json.loads(base_meta.read_text())
    metadata.update({"id": kernel_ref, "title": title, "code_file": code_name,
                     "enable_gpu": True, "enable_internet": False})
    metadata.pop("keywords", None)
    metadata["dataset_sources"] = sorted(set(metadata["dataset_sources"])
                                         | {dataset_ref, CACHE_REF})

    if experiment == "E42":
        code = next(c.source for c in nb.cells if c.cell_type == "code")
        code = code.replace(
            "from e13_b_inference import make_features",
            "from e13_b_inference import make_features" + FAMILY_MAX_SCORES_COMPAT)
        anchor = "pred=route_family_max(probs.argmax(2),scores,ensemble_probs,families)"
        if code.count(anchor) != 1:
            raise ValueError("E13-B family-max output anchor changed")
        addition = anchor + r'''
e13b_independent=pred.copy()
raw_scores,finite_scores=family_max_scores(probs.argmax(2),scores,ensemble_probs,families)
''' + E41_INFERENCE + r'''
e42_scores=blend_equal(finite_scores,e41_probs)
pred=np.argmax(e42_scores,axis=1)
'''
        bootstrap_at = "from src.inference import predict_candidate"
        if code.count(bootstrap_at) != 1:
            raise ValueError("E13-B runtime bootstrap anchor changed")
        code = code.replace(bootstrap_at, bootstrap_at + "\n" + E41_BOOTSTRAP)
        code = code.replace(anchor, addition)
        code = code.replace("'ensemble':'E13-B'", "'ensemble':'E42 E13-B plus E41'")
        code = code.replace("'runtime_seconds':round(time.time()-started,1)",
            "'runtime_seconds':round(time.time()-started,1),"
            "'e38_e39_seconds':round(e38_e39_seconds,1),"
            "'changed_from_e13b':int(np.count_nonzero(pred!=e13b_independent)),"
            "'member_weights':{'E13-B':0.5,'E41':0.5}")
        if "changed_from_e13b" not in code:
            raise ValueError("E42 run manifest update failed")
        for c in nb.cells:
            if c.cell_type == "code":
                c.source = code
                break
        nb.cells[0].source = "# E42 — E13-B family-max plus E41 equal probability member\n\nProduces one validated submission.parquet."
    elif experiment == "E43":
        code = nb.cells[2].source
        code = code.replace(
            "from e13_b_inference import make_features, family_max_scores",
            "from e13_b_inference import make_features" + FAMILY_MAX_SCORES_COMPAT)
        anchor = 'if not np.isfinite(e33_scores).all(): raise ValueError("E33 adjusted score matrix is nonfinite")'
        if code.count(anchor) != 1:
            raise ValueError("E33-F adjusted-score anchor changed")
        addition = anchor + "\ne33f_independent=independent.copy()\n" + E41_INFERENCE + r'''
e43_scores=blend_equal(e33_scores,e41_probs)
independent=np.argmax(e43_scores,axis=1)
'''
        bootstrap_at = "from src.e28_deploy import checkpoint_bundle_relative_path, distribute_checkpoints"
        if code.count(bootstrap_at) != 1:
            raise ValueError("E33-F runtime bootstrap anchor changed")
        code = code.replace(bootstrap_at, bootstrap_at + "\n" + E41_BOOTSTRAP)
        code = code.replace(anchor, addition)
        code = code.replace('write_submission("E33-F Independent","submission.parquet",independent)',
                            'write_submission("E43 E33-F plus E41","submission.parquet",independent)')
        code = code.replace('run={"experiment":"E31-40"', 'run={"experiment":"E43 E33-F plus E41"')
        code = code.replace('"inference_batch_sizes":effective_batches}',
            '"inference_batch_sizes":effective_batches,"e38_e39_seconds":round(e38_e39_seconds,1),'
            '"changed_from_e33f":int(np.count_nonzero(independent!=e33f_independent)),'
            '"member_weights":{"E33-F":0.5,"E41":0.5}}')
        nb.cells[2].source = code
        nb.cells[0].source = "# E43 — E33-F candidate scores plus E41 equal probability member\n\nProduces one validated submission.parquet."
    else:
        raise ValueError(experiment)

    nbformat.validate(nb)
    nbformat.write(nb, output_dir / code_name)
    (output_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def prepare() -> dict:
    if not (AUX / "e38_e39_manifest.json").is_file():
        raise FileNotFoundError(f"shared E38/E39 dataset is missing: {AUX}")
    if not (CACHE / "tiktoken_cache_manifest.json").is_file():
        raise FileNotFoundError(f"offline tiktoken cache is missing: {CACHE}")
    result = {}
    for experiment, (relative, kernel_ref, code_name) in OUTPUTS.items():
        kind = "e13" if experiment == "E42" else "e33"
        notebook_path, metadata_path = (ROOT / p for p in PULLS[kind])
        if not notebook_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"saved baseline Kaggle source is missing for {kind}")
        target = ROOT / relative
        title = ("E42 E13-B Plus E41" if experiment == "E42"
                 else "E43 E33-F Plus E41")
        _append_source(AUX_REF, title, target, kernel_ref, code_name,
                       notebook_path, metadata_path, experiment)
        embed_runtime_preflight(
            target,
            required_distributions=["catboost", "tiktoken", "transformers",
                                    "scikit-learn", "joblib", "numpy", "scipy"],
            # The runtime bootstrap installs the pickled feature class
            # before loading E38; asset checks still run in the gate.
            sklearn_compatibility_assets=[],
            manifest_paths=[AUX / "e38_e39_manifest.json",
                            CACHE / "tiktoken_cache_manifest.json"],
        )
        local = validate_kernel_package(target)
        result[experiment] = {"kernel_directory": str(target),
                              "kernel_ref": kernel_ref, "preflight": local}
    return result


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
