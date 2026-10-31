import unittest
import numpy as np
import pandas as pd
from src.data_factory import make_splits, validate_data
from src.evaluation import align_predictions, ensemble_probabilities, evaluate, complementarity


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


class EvaluationTests(unittest.TestCase):
    def test_metrics_and_probability_validation(self):
        y=["a","b","a"]
        p=np.array([[.9,.1],[.2,.8],[.4,.6]])
        result=evaluate(y,np.log(p),["a","b"],p)
        self.assertAlmostEqual(result["balanced_accuracy"],.75)
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


if __name__=="__main__":unittest.main()
