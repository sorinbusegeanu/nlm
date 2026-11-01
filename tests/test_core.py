import unittest
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from src.data_factory import make_splits, validate_data
from src.evaluation import align_predictions, ensemble_probabilities, evaluate, complementarity, paired_group_bootstrap
from src.feature_factory import FoldFeatures, load_or_fit
from src.ensemble_factory import cross_fitted_weights, fit_weights
from src.registry import Registry
from src.model_factory import train_svc
from src.pipeline import _load_oof, evaluate_ensembles, _model_cfg
from src.inference import export_svc_submission, package_offline_model
import joblib
from unittest.mock import patch


class DataTests(unittest.TestCase):
    def setUp(self):
        rows=[]
        for label in ("a","b","c"):
            for group in range(12): rows.append({"id":f"{label}{group}","prompt_id":f"p{group}","text":f"{label} text {group}","model":label})
        self.frame=pd.DataFrame(rows)

    def test_group_splits_are_disjoint_and_complete(self):
        assignments,folds=make_splits(self.frame,seed=7,n_splits=3)
        hold=set(assignments.loc[assignments.split.eq("holdout"),"prompt_id"])
        dev=set(assignments.loc[assignments.split.eq("development"),"prompt_id"])
        self.assertFalse(hold & dev)
        self.assertEqual(len(folds),len(dev)*3)
        for fold in range(3):
            self.assertEqual(set(folds.loc[folds.fold.eq(fold),"prompt_id"]) & set(folds.loc[folds.fold.ne(fold),"prompt_id"]),set())

    def test_validation_rejects_duplicate_ids(self):
        frame=self.frame.copy();frame.loc[1,"id"]=frame.loc[0,"id"]
        self.assertTrue(any("unique" in e for e in validate_data(frame)))

    def test_prepare_profile_is_repeatable_and_split_is_frozen(self):
        from src.data_factory import prepare
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/"train.csv";self.frame.to_csv(source,index=False)
            def fake_parquet(frame,path,index=False): Path(path).write_bytes(b"parquet test stub")
            with patch.object(pd.DataFrame,"to_parquet",fake_parquet):
                first=prepare(source,Path(td)/"out",seed=7,holdout_fraction=.2,n_splits=3,expected_classes=3)
                second=prepare(source,Path(td)/"out",seed=7,holdout_fraction=.2,n_splits=3,expected_classes=3)
                self.assertEqual(first["split_sha256"],second["split_sha256"])
                with self.assertRaises(FileExistsError):
                    prepare(source,Path(td)/"out",seed=8,holdout_fraction=.2,n_splits=3,expected_classes=3)


class EvaluationTests(unittest.TestCase):
    def test_metrics_and_probability_validation(self):
        y=["a","b","a"]
        p=np.array([[.9,.1],[.2,.8],[.7,.3]])
        result=evaluate(y,np.array([[2.,1.],[1.,2.],[1.,2.]]),["a","b"],p)
        self.assertAlmostEqual(result["balanced_accuracy"],.75)
        self.assertAlmostEqual(result["calibrated_balanced_accuracy"],1.0)
        self.assertIn("log_loss",result)

    def test_alignment_rejects_missing_and_duplicate_ids(self):
        a=pd.DataFrame({"id":[1,2],"a":[.8,.2],"b":[.2,.8]})
        b=pd.DataFrame({"id":[2,1],"a":[.3,.7],"b":[.7,.3]})
        joined=align_predictions({"a":a,"b":b},["a","b"])
        self.assertEqual(joined.id.tolist(),[1,2])
        with self.assertRaises(ValueError):align_predictions({"a":a,"b":b.iloc[:1]},["a","b"])

    def test_ensemble_and_complementarity(self):
        y=np.array(["a","b"]);p1=np.array([[.8,.2],[.2,.8]]);p2=p1[::-1]
        blend=ensemble_probabilities([p1,p2]);self.assertTrue(np.allclose(blend.sum(axis=1),1))
        self.assertEqual(complementarity(y,{"one":np.array(["a","b"]),"two":np.array(["b","a"])})["one__two"]["oracle_accuracy"],1)

    def test_group_bootstrap_is_deterministic(self):
        y=np.array(["a","b"]*6);a=np.array(["a","a"]*6);b=y.copy();groups=np.repeat(np.arange(6),2)
        first=paired_group_bootstrap(y,a,b,groups,n_bootstrap=100,seed=5)
        second=paired_group_bootstrap(y,a,b,groups,n_bootstrap=100,seed=5)
        self.assertEqual(first,second);self.assertGreater(first["delta_balanced_accuracy"],0)


class FeatureAndModelTests(unittest.TestCase):
    def test_fold_features_do_not_fit_on_validation_vocabulary(self):
        train=["trainwordalpha common", "trainwordbeta common"]
        validation=["validationonlytoken totallydifferent"]
        features=FoldFeatures({"views":["word"]});features.fit_transform(train);features.transform(validation)
        vocab=features.vectorizers["word"].vocabulary_
        self.assertNotIn("validationonlytoken",vocab)

    def test_feature_cache_changes_with_validation_rows(self):
        with tempfile.TemporaryDirectory() as td:
            _,_,_,key_a=load_or_fit(["train text alpha", "train text beta"],["valid text one"],config={"views":["word"]},cache_dir=td,fold_token="f0")
            _,_,_,key_b=load_or_fit(["train text alpha", "train text beta"],["different valid text"],config={"views":["word"]},cache_dir=td,fold_token="f0")
            self.assertNotEqual(key_a,key_b)

    def test_svc_returns_aligned_raw_and_calibrated_outputs(self):
        train_text=[];labels=[]
        for label,token in [("a","aardvark"),("b","bananas"),("c","copper")]:
            for i in range(12):train_text.append(f"{token} common language sample number {i}");labels.append(label)
        result=train_svc(train_text,labels,["aardvark common validation","bananas common validation"],["a","b","c"],calibration_cv=2)
        self.assertEqual(result["scores"].shape,(2,3));self.assertEqual(result["probabilities"].shape,(2,3))
        self.assertTrue(np.isfinite(result["scores"]).all());self.assertTrue(np.allclose(result["probabilities"].sum(1),1))

    def test_cross_fitted_weights_and_registry(self):
        y=np.array(["a","b"]*10);p1=np.eye(2)[np.arange(20)%2]*.8+.1;p2=np.eye(2)[np.arange(20)%2]*.6+.2
        folds=np.repeat(np.arange(5),4)
        from unittest.mock import patch
        with patch("src.ensemble_factory.fit_weights", wraps=fit_weights) as fit_mock:
            pred,weights=cross_fitted_weights([p1,p2],y,folds,["a","b"])
        self.assertEqual(len(fit_mock.call_args_list),5)
        self.assertTrue(all(len(call.args[0][0]) == len(y)-4 for call in fit_mock.call_args_list))
        self.assertEqual(pred.shape,p1.shape);self.assertTrue(np.isfinite(pred).all())
        for w in weights.values():self.assertAlmostEqual(sum(w),1,places=5)
        self.assertAlmostEqual(float(fit_weights([p1,p2],y,["a","b"]).sum()),1)
        with tempfile.TemporaryDirectory() as td:
            reg=Registry(Path(td)/"registry.sqlite");reg.add("e","svc","dev",{}, {"balanced_accuracy":.5},seed=2)
            self.assertEqual(reg.rows()[0]["seed"],2)

    def test_hashed_oof_files_align_and_feed_ensemble_cli(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);pred_dir=root/"predictions";pred_dir.mkdir()
            base=pd.DataFrame({"id":[f"i{i}" for i in range(8)],"prompt_id":[f"p{i//2}" for i in range(8)],
                "y_true":["a","b"]*4,"fold":[0,0,1,1,0,0,1,1],
                "prob::a":[.9,.1,.8,.2,.9,.1,.8,.2],"prob::b":[.1,.9,.2,.8,.1,.9,.2,.8]})
            other=base.copy();other["prob::a"]=other["prob::a"]*.95+.025;other["prob::b"]=1-other["prob::a"]
            def write_pickle(frame,path,index=False): frame.to_pickle(path)
            with patch.object(pd.DataFrame,"to_parquet",write_pickle),patch.object(pd,"read_parquet",lambda path,**kwargs:pd.read_pickle(path)):
                base.to_parquet(pred_dir/"svc_v0_datahash_oof.parquet",index=False)
                other.to_parquet(pred_dir/"qwen06_v0_datahash_oof.parquet",index=False)
                manifest={"classes":["a","b"],"dataset_sha256":"datahash","split_sha256":"splithash"}
                (pred_dir/"svc_v0_datahash_manifest.json").write_text(json.dumps(manifest))
                (pred_dir/"qwen06_v0_datahash_manifest.json").write_text(json.dumps(manifest))
                aligned=_load_oof("v0",root)
                self.assertEqual(set(aligned[5]),{"svc_v0_datahash","qwen06_v0_datahash"})
                (pred_dir/"qwen06_v0_datahash_manifest.json").write_text(json.dumps({**manifest,"split_sha256":"other"}))
                with self.assertRaisesRegex(ValueError,"different dataset or split hashes"):_load_oof("v0",root)
                (pred_dir/"qwen06_v0_datahash_manifest.json").write_text(json.dumps(manifest))
                report=evaluate_ensembles("v0",root,root/"registry.sqlite",bootstrap_replicates=10)
                self.assertIn("balanced_accuracy",report)

    def test_neural_model_config_requires_pinned_revision(self):
        with self.assertRaisesRegex(ValueError,"explicit checkpoint and immutable revision"):
            _model_cfg("qwen06")

    def test_offline_submission_preserves_ids_and_schema(self):
        train_text=[];labels=[]
        for label,token in [("a","aardvark"),("b","bananas"),("c","copper")]:
            for i in range(8):train_text.append(f"{token} common language sample number {i}");labels.append(label)
        result=train_svc(train_text,labels,["aardvark common"],["a","b","c"],calibration_cv=2)
        with tempfile.TemporaryDirectory() as td:
            checkpoint=Path(td)/"model.joblib"
            joblib.dump({"model":result["model"],"features":result["features"],"classes":["a","b","c"]},checkpoint)
            test=Path(td)/"test.csv";pd.DataFrame({"id":["002","001"],"text":["aardvark common","bananas common"]}).to_csv(test,index=False)
            out=Path(td)/"submission.csv";export_svc_submission(checkpoint,test,out)
            result_frame=pd.read_csv(out,dtype={"id":"string"})
            self.assertEqual(result_frame.columns.tolist(),["id","model"])
            self.assertEqual(result_frame.id.tolist(),["002","001"])
            self.assertTrue(set(result_frame.model).issubset({"a","b","c"}))
            manifest=Path(td)/"manifest.json"
            manifest.write_text(json.dumps({"model":"svc","checkpoint":str(checkpoint),"classes":["a","b","c"],"experiment_id":"test"}))
            package=Path(td)/"model_package.zip"
            package_offline_model(manifest,package)
            unpacked=Path(td)/"unpacked";unpacked.mkdir()
            import zipfile
            with zipfile.ZipFile(package) as archive:archive.extractall(unpacked)
            portable_out=Path(td)/"portable.csv"
            subprocess.run([sys.executable,"-c","from src.inference import export_submission; export_submission('svc','manifest.json',r'%s',r'%s')"%(test,portable_out)],cwd=unpacked,check=True,capture_output=True,text=True)
            self.assertEqual(pd.read_csv(portable_out,dtype={"id":"string"}).id.tolist(),["002","001"])

    def test_kaggle_notebook_is_valid_and_explicit_about_paths(self):
        notebook=json.loads(Path("notebooks/kaggle_inference.ipynb").read_text())
        source="\n".join("".join(cell.get("source",[])) for cell in notebook["cells"])
        self.assertEqual(notebook["nbformat"],4)
        self.assertIn("MODEL_PACKAGE",source);self.assertIn("TEST_CSV",source)
        self.assertIn("submission.parquet",source);self.assertIn("TRANSFORMERS_OFFLINE",source)


if __name__=="__main__":unittest.main()
