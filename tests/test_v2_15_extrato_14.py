import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data_factory import _normal_text
from src.v2_15_extrato_14 import _prompt_group, build_dataset


class V215DatasetTests(unittest.TestCase):
    def test_normalization_prompt_grouping_and_text_preservation(self):
        self.assertEqual(_normal_text("  Ａ  B\nC "), "a b c")
        self.assertEqual(_prompt_group("Same prompt", "source-a", "a"),
                         _prompt_group("same   prompt", "source-b", "b"))
        self.assertNotEqual(_prompt_group("", "source-a", "local-1"),
                            _prompt_group("", "source-b", "local-1"))

    def test_incremental_build_deduplicates_and_audits_all_source_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            out = Path(tmp) / "processed"
            for folder in ("benchmarks_proxy", "ntl_extra_merged", "maniceet_cached",
                           "paragraph_without_signature", "oleg_18_model_wide"):
                (root / folder).mkdir(parents=True)
            v214_path, v0_path = out / "dataset_v2.14.parquet", out / "dataset_v0.parquet"
            out.mkdir(parents=True)
            pd.DataFrame([
                {"id": "old1", "text": "Already present", "model": "qwen3.8-max", "prompt_id": "v214-p"},
                {"id": "old2", "text": "Other old response", "model": "gpt-5.5", "prompt_id": "v214-q"},
            ]).to_parquet(v214_path, index=False)
            pd.DataFrame([{"id": "v0-id", "prompt_id": "p081", "text": "Official response", "model": "qwen3.8-max"}]).to_parquet(v0_path, index=False)

            pd.DataFrame([
                {"run": "r", "bank": "b", "prompt_id": "g1", "model": "gpt-5.5", "model_slug": "openai/gpt-5.5", "request": "Prompt A", "text": "  Fresh text.  ", "setting": "default", "finish_reason": "stop"},
                {"run": "r", "bank": "b", "prompt_id": "g2", "model": "qwen3.8-max", "model_slug": "qwen/qwen3.8-max", "request": "Prompt B", "text": "Official response", "setting": "default", "finish_reason": "stop"},
                {"run": "r", "bank": "b", "prompt_id": "g3", "model": "not-a-competition-label", "model_slug": "invalid", "request": "Prompt C", "text": "Invalid attribution row", "setting": "default", "finish_reason": "stop"},
            ]).to_csv(root / "benchmarks_proxy/ntl_proxy_generations.csv", index=False)
            pd.DataFrame([
                {"text": "Fresh text.", "model": "gpt-5.5", "group": "g1", "src": "proxy", "setting": "", "clean": True},
                {"text": "Already present", "model": "qwen3.8-max", "group": "g2", "src": "cached", "setting": "", "clean": True},
            ]).to_csv(root / "ntl_extra_merged/ntl_extra_merged.csv", index=False)
            pd.DataFrame([{"text": "Already present", "model": "qwen3.8-max", "prompt_id": "p2", "source": "maniceet"}]).to_parquet(root / "maniceet_cached/maniceet_cached.parquet", index=False)
            pd.DataFrame([{"row_index": 0, "source": "maniceet", "prompt_id": "p2", "model": "qwen3.8-max", "provider": "OpenAI"}]).to_parquet(root / "maniceet_cached/maniceet_cached_provenance.parquet", index=False)

            pd.DataFrame([{"id": "para1", "response_id": "para-response", "prompt_id": "p081",
                           "text": "A prompt-overlap response", "model": "qwen3.8-max"}]).to_csv(root / "paragraph_without_signature/train.csv", index=False)
            pd.DataFrame([{"prompt_id": "p081", "reconstructed_prompt": "Prompt from V0"}]).to_csv(root / "paragraph_without_signature/prompts.csv", index=False)
            pd.DataFrame([{"id": "unverified", "response_id": "u1", "prompt_id": "p2", "text": "x", "model": "gemini-3.7-flash"}]).to_csv(root / "paragraph_without_signature/gemini37_unverified.csv", index=False)

            pd.DataFrame([{"id": "row1", "prompt": "A distinct prompt", "qwen3.8-max": "Oleg response"}]).to_parquet(root / "oleg_18_model_wide/external_all_wide.parquet", index=False)
            pd.DataFrame([{"view": "all", "id": "row1", "model": "qwen3.8-max", "response_id": "resp1",
                           "prompt_group_id": "exact:abc", "prompt_exact": True, "text_only_suitable": True,
                           "mechanically_eligible": True}]).to_parquet(root / "oleg_18_model_wide/external_cell_metadata.parquet", index=False)
            pd.DataFrame([{"response_id": "resp1", "model": "qwen3.8-max", "source": "source-z",
                           "source_url": "https://example.invalid/source", "source_row_id": "z1",
                           "model_evidence_class": "request_and_returned_model", "license": "CC BY 4.0"}]).to_parquet(root / "oleg_18_model_wide/external_response_metadata.parquet", index=False)
            pd.DataFrame([{"response_id": "resp1", "model": "qwen3.8-max", "source": "source-z",
                           "source_row_id": "z1", "source_url": "https://example.invalid/source",
                           "license": "CC BY 4.0", "model_evidence_class": "request_and_returned_model",
                           "row_id_all": "row1", "prompt_group_id": "exact:abc",
                           "grouping_is_exact_user_prompt": True}]).to_parquet(root / "oleg_18_model_wide/external_cell_provenance.parquet", index=False)

            built = build_dataset(root=root, output_dir=out, v214_path=v214_path, v0_path=v0_path)
            data, audit = built["data"], built["profile"]["source_audit"]
            self.assertEqual(len(data), 2)
            self.assertIn("  Fresh text.  ", set(data.text))
            self.assertEqual(data.text.map(_normal_text).nunique(), len(data))
            self.assertEqual(set(data.model), {"gpt-5.5", "qwen3.8-max"})
            self.assertFalse(set(data.prompt_id) & {"p081"})
            self.assertEqual(audit["maniceet_cached"]["excluded"]["already_in_v2.14"], 1)
            self.assertEqual(audit["benchmarks_proxy"]["excluded"]["v0_id_or_response_overlap"], 1)
            self.assertEqual(audit["benchmarks_proxy"]["excluded"]["invalid_label"], 1)
            self.assertEqual(audit["paragraph_without_signature"]["excluded"]["v0_prompt_group_overlap"], 1)
            self.assertEqual(audit["paragraph_without_signature"]["excluded"]["unverifiable_model_attribution"], 1)
            for source in audit.values():
                self.assertEqual(source["input_rows"], source["added_rows"] + sum(source["excluded"].values()))


if __name__ == "__main__":
    unittest.main()
