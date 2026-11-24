"""Regenerate E33-F submissions locally from the completed E31 worker caches.

This recovery path does not launch or execute a Kaggle kernel. The checkpoint
probabilities were produced by E33 kernel v3 using the attached E31 checkpoint
dataset, then persisted before that kernel hit the missing ``src.e33`` import.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from transformers import AutoModel, AutoModelForMultimodalLM, AutoTokenizer

from src import e19b_group, e27, e33, e19f
from src.e13_b_inference import make_features, family_max_scores, route_family_max
from src.e31_deploy import cluster_f_candidates, hungarian
from src.e19b_reconstruct import _generation_inputs, parse_reconstruction
from src.e28_deploy import ensure_binary_logistic_predict_proba_compat

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/tmp/e33-v3-error/e31_probabilities")
WORK = Path("/tmp/e33-v3-error/e31_runtime_root")
E31_DIR = ROOT / "artifacts/kaggle/e31-checkpoints-v001"
E33_DIR = ROOT / "artifacts/kaggle/e33f-deployment-v001"
E19D_DIR = ROOT / "artifacts/analysis/e28"
OUT = ROOT / "artifacts/kaggle/e33-local-submissions-v3-cache"
QWEN = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
GEMMA = Path.home() / ".cache/huggingface/hub/models--google--embeddinggemma-2/snapshots/f9972ac22e696a105c482202646979e33e4b5a1e"


class TextOnlyQwenProcessor:
    """Match the Kaggle text-only reconstruction processor adapter."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(self, conversations, **kwargs):
        processor_kwargs = kwargs.pop("processor_kwargs", {}) or {}
        for key, value in processor_kwargs.items():
            kwargs.setdefault(key, value)
        return self.tokenizer.apply_chat_template(conversations, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def embed(texts: list[str], *, prefix: str = "task: clustering | query: ") -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(GEMMA, local_files_only=True, padding_side="right")
    model = AutoModel.from_pretrained(GEMMA, local_files_only=True,
                                      torch_dtype=torch.float16).to("cuda:0").eval()
    batches = []
    for start in range(0, len(texts), 16):
        batch = [prefix + str(x) for x in texts[start:start + 16]]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512,
                           return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state.float()
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        batches.append(pooled.cpu().numpy().astype(np.float32))
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    vectors = np.concatenate(batches)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-12)
    return vectors


def main() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Local E33 grouping recovery requires the local CUDA GPU")
    manifest = json.loads((E31_DIR / "e31_manifest.json").read_text())
    classes = manifest["classes"]
    members = manifest["members"]
    ids = pd.read_csv(ROOT / "data/raw/test.csv", dtype={"id": "string"})
    expected_texts = ids.text.astype(str).tolist()
    texts = json.loads((WORK / "texts.json").read_text())
    if texts != expected_texts:
        raise ValueError("Cached checkpoint probabilities were generated for different test rows")
    if json.loads((WORK / "classes.json").read_text()) != classes:
        raise ValueError("Cached probability class order differs from the E31 checkpoint manifest")
    worker = json.loads((ROOT / "/tmp/e33-v3-error/e31_worker_report.json").read_text())
    if not worker.get("parallel_overlap_verified") or len(worker.get("records", [])) != 40:
        raise ValueError("E33 v3 did not finish all 40 checkpoint workers concurrently")
    roster = [m["candidate_id"] for m in members]
    if {r["candidate_id"] for r in worker["records"]} != set(roster):
        raise ValueError("Completed worker report does not match the fixed 40-checkpoint roster")
    probs = np.stack([np.load(CACHE / f"{sid}.npz", allow_pickle=False)["probabilities"]
                      for sid in roster]).astype(np.float64)
    if probs.shape != (40, len(texts), 18) or not np.isfinite(probs).all():
        raise ValueError(f"Invalid checkpoint cache shape/data: {probs.shape}")

    deployment = manifest["deployment"]
    router = CatBoostClassifier()
    router.load_model(str(E31_DIR / "e31_router.cbm"))
    reliability = router.predict_proba(make_features(probs, members, classes,
                                                       deployment["reliability_statistics"]))[:, 1]
    reliability = reliability.reshape(len(texts), len(members)).T
    families = np.asarray([m["family"] for m in members])
    family_names = sorted(set(families))
    ensemble_probs = np.mean([probs[families == family].mean(0) for family in family_names], axis=0)
    ensemble_probs /= ensemble_probs.sum(1, keepdims=True)
    hard = probs.argmax(2)
    raw_scores, finite_scores = family_max_scores(hard, reliability, ensemble_probs, families)
    independent = route_family_max(hard, reliability, ensemble_probs, families)
    if not np.isfinite(finite_scores).all():
        raise ValueError("E31 deployment scores are non-finite")

    e33._BUILD_CP_IDS = roster
    schema = json.loads((E33_DIR / "e33_final_feature_schema.json").read_text())
    with np.load(E33_DIR / "e33_final_reliability.npz", allow_pickle=False) as z:
        reliability_stats = {k: z[k] for k in z.files}
    ranks = e33._rank_positions(probs)
    candidate_features = e33._build_candidate_features(np.arange(len(texts), dtype=np.int64),
        reliability_stats, probabilities=probs, rank_position=ranks, base_scores=finite_scores,
        families=families.tolist()).loc[:, schema["feature_order"]]
    e33_model = CatBoostClassifier()
    e33_model.load_model(str(E33_DIR / "e33_final.cbm"))
    wrong_class = schema["model_class_order"].index(1)
    p_correct = (1.0 - e33_model.predict_proba(candidate_features)[:, wrong_class]).reshape(len(texts), 18)
    e33_scores = finite_scores * p_correct
    independent = np.argmax(e33_scores, axis=1)

    # E19-F-A/F-C, identical frozen E31 grouping algorithms and thresholds.
    grouper_path = E31_DIR / "e19_f_c_external_grouper.joblib"
    grouper = joblib.load(grouper_path)
    ensure_binary_logistic_predict_proba_compat(grouper["pair_model"]["model"])
    emb = embed(texts)
    order = np.argsort(ids.id.astype(str).to_numpy(), kind="stable")
    pairs = np.asarray([(a, b) for a in range(len(texts)) for b in range(a + 1, len(texts))], dtype=np.int64).reshape(-1, 2)
    lexical = e19f._transform_lexical(np.asarray(texts, dtype=str), grouper["vectorizers"])
    pair_features = e19f._pair_features(order, pairs, lexical, emb)
    groups_a = cluster_f_candidates("E19-F-A", ids.id.astype(str).to_numpy(), order, pairs,
                                     pair_features, {"threshold": 0.85})
    groups_c = cluster_f_candidates("E19-F-C", ids.id.astype(str).to_numpy(), order, pairs,
                                     pair_features, grouper)

    # E19-D uses the same Qwen reconstruction and external pair scorer shipped
    # in E33's package. Generation is local; no Kaggle kernel is launched.
    processor = TextOnlyQwenProcessor(AutoTokenizer.from_pretrained(QWEN, local_files_only=True))
    qwen = AutoModelForMultimodalLM.from_pretrained(QWEN, device_map="auto", dtype="auto",
                                                     local_files_only=True).eval()
    inputs = _generation_inputs(processor, texts)
    target_device = next(qwen.parameters()).device
    inputs = {k: (v.to(target_device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    prefix_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        generated = qwen.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=1024,
            use_cache=True, pad_token_id=processor.tokenizer.eos_token_id)
    raw = processor.batch_decode(generated[:, prefix_len:], skip_special_tokens=True)
    reconstructed = []
    topics, details, constraints = [], [], []
    for i, value in enumerate(raw):
        try:
            item = parse_reconstruction(value)
        except Exception:
            item = {"reconstructed_prompt": texts[i][:4000], "topic": "",
                    "distinctive_details": [], "constraints": []}
        reconstructed.append(item["reconstructed_prompt"])
        topics.append(item.get("topic", "")); details.append(item.get("distinctive_details", []))
        constraints.append(item.get("constraints", []))
    del qwen, processor
    gc.collect(); torch.cuda.empty_cache()
    d_grouper = joblib.load(E19D_DIR / "e19_d_external_grouper.joblib")
    d_vectors = {name: d_grouper["vectorizers"][name] for name in
                 ("reconstruction_char", "reconstruction_word", "response_char", "response_word")}
    d_spaces = e19b_group.transform_similarity_spaces(reconstructed, texts, d_vectors)
    d_semantic = embed(reconstructed)
    d_rows, d_pairs, d_x = e27._reconstruction_features(np.arange(len(texts)), d_spaces,
        reconstructed, texts, d_semantic, topics, details, constraints)
    groups_d, _, _ = e27._pair_cluster(ids.id.astype(str).to_numpy()[d_rows], d_pairs, d_x,
        d_grouper["pair_scorer"], float(d_grouper["threshold"]))
    full_groups_d = np.empty(len(texts), dtype=object)
    full_groups_d[d_rows] = groups_d

    OUT.mkdir(parents=True, exist_ok=True)
    variants = {
        "E33-F Independent": independent,
        "E33-F E19-F-A": hungarian(e33_scores, groups_a, ids.id.astype(str).to_numpy()),
        "E33-F E19-F-C": hungarian(e33_scores, groups_c, ids.id.astype(str).to_numpy()),
        "E33-F E19-D": hungarian(e33_scores, full_groups_d, ids.id.astype(str).to_numpy()),
    }
    rows = {}
    for name, pred in variants.items():
        out = OUT / ("submission.parquet" if name.endswith("Independent") else
                     "submission_e19_f_a.parquet" if name.endswith("F-A") else
                     "submission_e19_f_c.parquet" if name.endswith("F-C") else "submission_e19_d.parquet")
        frame = pd.DataFrame({"id": ids.id.astype(str), "model": np.asarray(classes, dtype=object)[pred]})
        if frame.id.tolist() != ids.id.astype(str).tolist() or list(frame.columns) != ["id", "model"]:
            raise ValueError(f"Invalid submission schema/order for {name}")
        if frame.id.duplicated().any() or frame.model.isna().any() or not frame.model.isin(classes).all():
            raise ValueError(f"Invalid IDs/labels for {name}")
        frame.to_parquet(out, index=False)
        if not pd.read_parquet(out).equals(frame):
            raise ValueError(f"Parquet round-trip mismatch for {name}")
        rows[name] = {"file": out.name, "rows": len(frame), "sha256": sha(out),
                      "changed_vs_independent": int(np.count_nonzero(pred != independent))}
    report = {"source_kernel": "zodrak/e33-f-e31-40-checkpoint-candidate-router v3 (failed after E31 inference)",
        "checkpoint_dataset": "zodrak/e31-checkpoints-v001", "checkpoint_roster": roster,
        "checkpoint_cache_dir": str(CACHE), "test_ids": ids.id.astype(str).tolist(),
        "class_order": classes, "local_groupers": "E19-F-A, E19-F-C and E19-D frozen artifacts",
        "variants": rows, "kaggle_kernel_launched": False}
    (OUT / "recovery_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
