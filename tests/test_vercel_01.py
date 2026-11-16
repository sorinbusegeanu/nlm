import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src import vercel_01 as v


def _response(text="A useful answer in English.", model="Qwen/Qwen3-0.6B"):
    return {"model": model, "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 6,
                      "completion_tokens_details": {"reasoning_tokens": 2}}}


def test_prompt_csv_validation_optional_metadata_and_flags(tmp_path):
    path = tmp_path / "prompts.csv"
    path.write_text("prompt_id,prompt_text,category\nP2,Explain why stars twinkle.,science\nP1,Write a Python function.,code\n", encoding="utf8")
    frame, digest, violations = v.load_prompts(path)
    assert frame.prompt_id.tolist() == ["P1", "P2"]
    assert frame.loc[0, "category"] == "code"
    assert len(digest) == 64
    assert violations.prompt_id.tolist() == ["P1"]
    path.write_text("prompt_id,prompt_text\nP1,First prompt\nP1,Second prompt\n", encoding="utf8")
    with pytest.raises(ValueError, match="unique"):
        v.load_prompts(path)


def test_plan_reports_violations_without_editing_prompt_library(tmp_path, capsys):
    path = tmp_path / "prompts.csv"
    original = "prompt_id,prompt_text\nP1,Explain why stars twinkle in the night sky.\nP2,Write Python code for sorting.\n"
    path.write_text(original, encoding="utf8")
    result = v.plan(path, "local", ["local-qwen"], limit=None)
    assert result["prompt_violations"] == 1
    assert result["prompt_violations_by_flag"]["coding_or_programming_prompt"] == 1
    assert result["eligible_prompts"] == 1
    assert path.read_text(encoding="utf8") == original
    capsys.readouterr()


def test_reasoning_assignment_and_model_adapter_are_deterministic():
    mode_map = v.assign_reasoning_modes([f"P{i:05d}" for i in range(5000)])
    modes = list(mode_map.values())
    assert sum(mode == "high" for mode in modes) == 1500
    assert mode_map == v.assign_reasoning_modes([f"P{i:05d}" for i in range(5000)])
    model = {"label": "test", "request_parameters": {"temperature": 0},
             "reasoning_supported": True, "high_effort_parameters": {"reasoning_effort": "high"}}
    assert v.request_parameters(model, "default") == {"temperature": 0}
    assert v.request_parameters(model, "high") == {"temperature": 0, "reasoning_effort": "high"}
    assert v.request_parameters({"reasoning_supported": False}, "high") == {}


def test_cost_reservation_requires_prices_and_uses_completion_cap():
    model = {"label": "cheap", "max_completion_tokens": 1000,
             "pricing": {"input_usd_per_million": 2, "output_usd_per_million": 10}}
    assert v.request_cost_bound(model, "Question?") >= .01
    with pytest.raises(v.ConfigurationError):
        v.request_cost_bound({"label": "missing", "max_completion_tokens": 50}, "question")


def test_local_generation_persists_and_exports_with_optional_columns(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts.csv"
    prompts.write_text("prompt_id,prompt_text,category\nP1,Explain why stars twinkle in the night sky.,astronomy\n", encoding="utf8")
    root, processed = tmp_path / "runs", tmp_path / "processed"
    monkeypatch.setattr(v, "ARTIFACT_ROOT", root)
    monkeypatch.setattr(v, "PROCESSED", processed)
    monkeypatch.setattr(v, "_maybe_register_dataset", lambda *args: False)
    monkeypatch.setenv("VLLM_API_KEY", "mock-key")
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            assert kwargs["messages"][-1]["role"] == "user"
            assert len(kwargs["messages"]) == 1
            return _response()

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(v, "OpenAI", FakeClient)
    result = v.generate(prompts, backend="local", model_names=["local-qwen"], limit=1)
    assert result["status"] == "completed"
    journal = root / result["run_id"] / "responses.jsonl"
    row = json.loads(journal.read_text().strip())
    assert row["id"] == v.stable_response_id("vercel_local_test", "P1", "local-qwen")
    assert row["input_tokens"] == 9 and row["reasoning_tokens"] == 2
    data = pd.read_parquet(processed / "dataset_vercel_local_test.parquet")
    assert {"id", "prompt_id", "text", "model", "prompt_category"}.issubset(data.columns)
    assert calls[0]["model"] == "Qwen/Qwen3-0.6B"
    v.generate(resume=result["run_id"])
    assert len(calls) == 1


def test_long_response_is_journaled_but_rejected_from_parquet(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts.csv"
    prompts.write_text("prompt_id,prompt_text\nP1,Explain why stars twinkle in the night sky.\n", encoding="utf8")
    monkeypatch.setattr(v, "ARTIFACT_ROOT", tmp_path / "runs")
    monkeypatch.setattr(v, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(v, "_maybe_register_dataset", lambda *args: False)
    monkeypatch.setenv("VLLM_API_KEY", "mock")

    class FakeCompletions:
        def create(self, **kwargs):
            return _response("word " * 401)

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(v, "OpenAI", FakeClient)
    result = v.generate(prompts, backend="local", model_names=["local-qwen"])
    row = json.loads((tmp_path / "runs" / result["run_id"] / "responses.jsonl").read_text())
    assert row["accepted"] is False
    assert row["exclusion_reason"] == "over_2000_characters"
    assert len(row["text"]) > 2000
    assert len(pd.read_parquet(tmp_path / "processed/dataset_vercel_local_test.parquet")) == 0
    assert (tmp_path / "runs" / result["run_id"] / "rejected_responses.csv").exists()


def test_retry_after_and_five_attempt_limit_survive_resume(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts.csv"
    prompts.write_text("prompt_id,prompt_text\nP1,Explain why stars twinkle in the night sky.\n", encoding="utf8")
    monkeypatch.setattr(v, "ARTIFACT_ROOT", tmp_path / "runs")
    monkeypatch.setattr(v, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(v, "_maybe_register_dataset", lambda *args: False)
    monkeypatch.setenv("VLLM_API_KEY", "mock")
    attempts = []

    class TooManyRequests(Exception):
        status_code = 429
        response = SimpleNamespace(headers={"Retry-After": "0"})

    class FakeCompletions:
        def create(self, **kwargs):
            attempts.append(1)
            raise TooManyRequests("rate limited")

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(v, "OpenAI", FakeClient)
    monkeypatch.setattr(v.RateLimiter, "wait", lambda self, *args: None)
    result = v.generate(prompts, backend="local", model_names=["local-qwen"])
    assert result["status"] == "failed"
    assert len(attempts) == 5
    v.generate(resume=result["run_id"])
    assert len(attempts) == 5
    journal = tmp_path / "runs" / result["run_id"] / "attempts.jsonl"
    events = v.read_jsonl(journal)
    assert len(events) == 5
    assert sum(row["status"] in {"retry", "failed"} for row in events) == 5
    assert all(row["error_classification"] == "transient" for row in events)
    budget_events = v.read_jsonl(tmp_path / "runs" / result["run_id"] / "budget_events.jsonl")
    assert sum(row["event"] == "reserved" for row in budget_events) == 5


def test_nonmatching_returned_model_is_rejected(tmp_path):
    model = {"label": "test", "model_id": "asked/model", "returned_model_ids": [],
             "max_completion_tokens": 20, "pricing": {"input_usd_per_million": 1,
             "output_usd_per_million": 1}, "reasoning_supported": False}
    config = {"dataset_version": "vercel_v01", "max_response_characters": 2000,
              "retry": {"max_attempts": 5}, "budget": {"enforce": False},
              "_backend": "local", "system_prompt": ""}
    class Completion:
        def create(self, **kwargs):
            return _response(model="some/other-model")
    client = SimpleNamespace(chat=SimpleNamespace(completions=Completion()))
    with pytest.raises(v.ConfigurationError, match="identity mismatch"):
        v._call_one(client, {"prompt_id": "P1", "prompt_text": "A complete English sentence here."},
                    "test", model, "default", config, "run", tmp_path, SimpleNamespace(wait=lambda *args: None),
                    {"spent": 0, "reserved": 0})
    assert not (tmp_path / "responses.jsonl").exists()


def test_budget_blocks_request_before_call(tmp_path):
    model = {"label": "test", "model_id": "asked/model", "max_completion_tokens": 2048,
             "pricing": {"input_usd_per_million": 10, "output_usd_per_million": 50},
             "reasoning_supported": False}
    config = {"dataset_version": "vercel_v01", "max_response_characters": 2000,
              "retry": {"max_attempts": 5}, "budget": {"enforce": True, "max_spend_usd": .001},
              "_backend": "vercel", "system_prompt": ""}
    class Completion:
        def create(self, **kwargs):
            raise AssertionError("budget must block before API request")
    client = SimpleNamespace(chat=SimpleNamespace(completions=Completion()))
    with pytest.raises(v.BudgetExhausted):
        v._call_one(client, {"prompt_id": "P1", "prompt_text": "A complete English sentence here."},
                    "test", model, "default", config, "run", tmp_path, SimpleNamespace(wait=lambda: None),
                    {"spent": 0, "reserved": 0})


def test_production_budget_exhaustion_and_resume_override(tmp_path, monkeypatch):
    import yaml
    prompts = tmp_path / "prompts.csv"
    prompts.write_text("prompt_id,prompt_text\nP1,Explain why stars twinkle in the night sky.\n", encoding="utf8")
    config = v.load_config()
    config["budget"]["max_spend_usd"] = 0.0001
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf8")
    runs, processed = tmp_path / "runs", tmp_path / "processed"
    monkeypatch.setattr(v, "ARTIFACT_ROOT", runs)
    monkeypatch.setattr(v, "PROCESSED", processed)
    monkeypatch.setattr(v, "_maybe_register_dataset", lambda *args: False)
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "mock")
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return {"model": kwargs["model"], "choices": [{"message": {"content": "An answer."}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 4}}

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(v, "OpenAI", FakeClient)
    monkeypatch.setattr(v.RateLimiter, "wait", lambda self, *args: None)
    stopped = v.generate(prompts, backend="vercel", model_names=["gpt-5.6-luna"], config_path=config_path)
    assert stopped["status"] == "budget_exhausted"
    assert calls == []
    resumed = v.generate(resume=stopped["run_id"], budget_override=1.0)
    assert resumed["status"] == "completed"
    assert len(calls) == 1
    manifest = json.loads((runs / stopped["run_id"] / "run_manifest.json").read_text())
    assert manifest["budget_limit_usd"] == 1.0
    assert manifest["budget_spent_estimated_usd"] < 1.0
    profile = json.loads((processed / "dataset_profile_vercel_v01.json").read_text())
    assert profile["estimated_cost_usd"] > 0
    assert profile["model_statistics"]["gpt-5.6-luna"]["estimated_cost_usd"] > 0


def test_backend_config_and_catalog_class_gate(monkeypatch):
    config = v.load_config()
    monkeypatch.setenv("VLLM_API_KEY", "local-key")
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "gateway-key")
    local = v._client_config(config, "local")
    gateway = v._client_config(config, "vercel")
    assert local[:2] == ("http://localhost:8000/v1", "local-key")
    assert gateway[:2] == ("https://ai-gateway.vercel.sh/v1", "gateway-key")
    assert len(config["models"]) == 18
    assert v._maybe_register_dataset(Path("missing"), Path("missing"), {"classes": ["gpt-5.5"]}) is False
