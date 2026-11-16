"""Resumable, budget-limited response generation for the Vercel v01 dataset.

This module deliberately separates generation from classifier training. The JSONL
response journal is the durable source of truth; Parquet and profile files are
rebuildable exports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import tempfile
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml
from langdetect import DetectorFactory, LangDetectException, detect
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/vercel_01.yaml"
ARTIFACT_ROOT = ROOT / "artifacts/vercel_01"
PROCESSED = ROOT / "data/processed"
LABELS = (
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "glm-5.3", "glm-5.3-flash",
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "kimi-k3", "qwen3.8-max",
)
TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
_journal_lock = threading.Lock()
_budget_lock = threading.Lock()


class ConfigurationError(ValueError):
    """Invalid generation configuration or non-retryable provider settings."""


class BudgetExhausted(RuntimeError):
    pass


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_response_id(dataset_version: str, prompt_id: str, label: str) -> str:
    raw = f"{dataset_version}\0{prompt_id}\0{label}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ConfigurationError("configuration must be a YAML mapping")
    if not re.fullmatch(r"vercel_v\d{2,}|vercel_local_test", str(config.get("dataset_version", ""))):
        raise ConfigurationError("dataset_version must be vercel_vNN or vercel_local_test")
    if not isinstance(config.get("models"), dict):
        raise ConfigurationError("configuration must include models")
    if float(config.get("budget", {}).get("max_spend_usd", 0)) <= 0:
        raise ConfigurationError("budget.max_spend_usd must be positive")
    if int(config.get("throttling", {}).get("concurrency", 0)) < 1:
        raise ConfigurationError("throttling.concurrency must be at least 1")
    if float(config.get("throttling", {}).get("requests_per_minute", 0)) <= 0:
        raise ConfigurationError("throttling.requests_per_minute must be positive")
    attempts = int(config.get("retry", {}).get("max_attempts", 0))
    if not 1 <= attempts <= 5:
        raise ConfigurationError("retry.max_attempts must be between 1 and 5")
    if int(config.get("request_timeout_seconds", 0)) <= 0:
        raise ConfigurationError("request_timeout_seconds must be positive")
    if set(config["models"]) != set(LABELS):
        raise ConfigurationError("the Gateway model registry must define exactly the 18 canonical labels")
    for label in LABELS:
        model = config["models"].get(label)
        if not isinstance(model, dict) or not model.get("model_id"):
            raise ConfigurationError(f"missing exact Gateway model ID for {label}")
        if not model.get("default_reasoning"):
            raise ConfigurationError(f"{label}: default reasoning behavior must be explicit in configuration")
        if not isinstance(model.get("max_completion_tokens"), int) or model["max_completion_tokens"] <= 0:
            raise ConfigurationError(f"{label}: max_completion_tokens must be a positive integer")
        if model.get("requests_per_minute") is not None and float(model["requests_per_minute"]) <= 0:
            raise ConfigurationError(f"{label}: requests_per_minute must be positive")
        if model.get("tokens_per_minute") is not None and int(model["tokens_per_minute"]) <= 0:
            raise ConfigurationError(f"{label}: tokens_per_minute must be positive")
        price = model.get("pricing", {})
        if (price.get("input_usd_per_million") is None or price.get("output_usd_per_million") is None or
                not price.get("source_url") or not price.get("checked_at")):
            raise ConfigurationError(f"{label}: pricing rates, source_url, and checked_at are required")
        request_parameters(model, "default")
        if model.get("reasoning_supported"):
            request_parameters(model, "high")
    return config


def load_prompts(path: str | Path) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    path = Path(path)
    digest = sha256_file(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    missing = {"prompt_id", "prompt_text"} - set(frame.columns)
    if missing:
        raise ValueError(f"prompt CSV requires columns: {', '.join(sorted(missing))}")
    frame["prompt_id"] = frame.prompt_id.astype(str).str.strip()
    frame["prompt_text"] = frame.prompt_text.astype(str)
    if frame.prompt_id.eq("").any() or frame.prompt_text.str.strip().eq("").any():
        raise ValueError("prompt_id and prompt_text must be nonempty")
    if frame.prompt_id.duplicated().any():
        duplicates = frame.loc[frame.prompt_id.duplicated(False), "prompt_id"].tolist()[:10]
        raise ValueError(f"prompt_id values must be unique; duplicates include {duplicates}")
    frame = frame.sort_values("prompt_id", kind="stable").reset_index(drop=True)
    violations = []
    for row in frame.itertuples(index=False):
        flags = prompt_flags(row.prompt_text)
        if flags:
            violations.append({"prompt_id": row.prompt_id, "flags": ";".join(flags),
                               "prompt_text": row.prompt_text})
    return frame, digest, pd.DataFrame(violations, columns=["prompt_id", "flags", "prompt_text"])


_CODE_RE = re.compile(r"\b(python|javascript|typescript|sql|html|css|program|programming|code|function|script|debug|compile|algorithm)\b|```", re.I)


def prompt_flags(text: str) -> list[str]:
    flags = []
    DetectorFactory.seed = 2026
    try:
        if detect(text) != "en":
            flags.append("non_english_prompt")
    except LangDetectException:
        flags.append("undetermined_language")
    if _CODE_RE.search(text):
        flags.append("coding_or_programming_prompt")
    return flags


def reasoning_mode(prompt_id: str, seed: int = 2026) -> str:
    """Return a stable 30% Bernoulli assignment for a single prompt ID."""
    number = int(hashlib.sha256(f"{seed}\0{prompt_id}".encode()).hexdigest()[:16], 16)
    return "high" if number % 10 < 3 else "default"


def assign_reasoning_modes(prompt_ids: Iterable[str], seed: int = 2026,
                           high_fraction: float = 0.30) -> dict[str, str]:
    """Assign an exact rounded share, stable for a given prompt-library set."""
    ids = list(dict.fromkeys(str(value) for value in prompt_ids))
    high_count = int(round(len(ids) * high_fraction))
    ranked = sorted(ids, key=lambda value: (hashlib.sha256(
        f"{seed}\0{value}".encode()).hexdigest(), value))
    high = set(ranked[:high_count])
    return {prompt_id: "high" if prompt_id in high else "default" for prompt_id in ids}


def request_parameters(model: dict[str, Any], mode: str) -> dict[str, Any]:
    """Build supported parameters; default mode intentionally omits effort."""
    params = dict(model.get("request_parameters", {}))
    if mode == "high":
        if not model.get("reasoning_supported", False):
            return params
        high = model.get("high_effort_parameters")
        if not isinstance(high, dict) or not high:
            raise ConfigurationError(f"{model.get('label', 'model')} has no high-effort adapter")
        params.update(high)
    allowed = {"temperature", "top_p", "max_tokens", "max_completion_tokens", "seed", "stop",
               "n", "reasoning_effort", "extra_body", "response_format", "frequency_penalty",
               "presence_penalty"}
    unknown = set(params) - allowed
    if unknown:
        raise ConfigurationError(f"unsupported Chat Completions parameters for {model.get('label')}: {sorted(unknown)}")
    return params


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return value
    return json.loads(json.dumps(value, default=str))


def parse_completion(response: Any) -> dict[str, Any]:
    raw = _jsonable(response)
    choices = raw.get("choices") or []
    if not choices:
        raise ValueError("Chat Completions response contains no choices")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    usage = raw.get("usage") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return {
        "text": str(content or ""), "returned_model_id": raw.get("model"),
        "finish_reason": choices[0].get("finish_reason"),
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "reasoning_tokens": int(completion_details.get("reasoning_tokens") or 0),
        "raw_response_metadata": raw,
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    with _journal_lock:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"corrupt JSONL at {path}:{lineno}") from exc
    return rows


def _prompt_tokens_upper_bound(text: str) -> int:
    # Four bytes per token is a conservative approximation for ordinary English,
    # with a fixed Chat Completions framing allowance.
    return max(1, (len(text.encode("utf-8")) + 2) // 3 + 16)


def request_cost_bound(model: dict[str, Any], prompt: str, system_prompt: str = "",
                       developer_prompt: str = "") -> float:
    price = model.get("pricing", {})
    if price.get("input_usd_per_million") is None or price.get("output_usd_per_million") is None:
        raise ConfigurationError(f"missing pricing for {model.get('label', model.get('model_id'))}")
    in_tokens = _prompt_tokens_upper_bound("\n".join(
        value for value in (system_prompt, developer_prompt, prompt) if value))
    out_tokens = int(model.get("max_completion_tokens", 0))
    return (in_tokens * float(price["input_usd_per_million"]) +
            out_tokens * float(price["output_usd_per_million"])) / 1_000_000


def _validate_tpm_limits(models: dict[str, dict[str, Any]], prompts: pd.DataFrame,
                         global_tpm: int | None, system_prompt: str = "",
                         developer_prompt: str = "") -> None:
    for label, model in models.items():
        tpm = model.get("tokens_per_minute", global_tpm)
        if not tpm:
            continue
        maximum = int(model.get("max_completion_tokens", 0)) + max(
            (_prompt_tokens_upper_bound("\n".join(value for value in
                (system_prompt, developer_prompt, text) if value)) for text in prompts.prompt_text.astype(str)), default=0)
        if maximum > int(tpm):
            raise ConfigurationError(
                f"{label}: one prompt may require {maximum} tokens, above configured {tpm} TPM limit"
            )


def _actual_cost(model: dict[str, Any], parsed: dict[str, Any]) -> float:
    price = model["pricing"]
    return (parsed["input_tokens"] * float(price["input_usd_per_million"]) +
            parsed["output_tokens"] * float(price["output_usd_per_million"])) / 1_000_000


class RateLimiter:
    def __init__(self, rpm: float, tpm: int | None = None):
        self.interval = 60.0 / rpm if rpm > 0 else 0
        self.tpm = int(tpm) if tpm else None
        self.lock = threading.Lock()
        self.next_request = 0.0
        self.token_events: deque[tuple[float, int]] = deque()

    def wait(self, estimated_tokens: int = 0) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                delay = max(0.0, self.next_request - now)
                if self.tpm and estimated_tokens > self.tpm:
                    raise ConfigurationError(f"single request estimate {estimated_tokens} exceeds TPM limit {self.tpm}")
                while self.token_events and now - self.token_events[0][0] >= 60:
                    self.token_events.popleft()
                token_sum = sum(item[1] for item in self.token_events)
                if self.tpm and token_sum + estimated_tokens > self.tpm and self.token_events:
                    delay = max(delay, 60 - (now - self.token_events[0][0]))
                if delay <= 0:
                    self.next_request = max(now, self.next_request) + self.interval
                    if self.tpm:
                        self.token_events.append((now, estimated_tokens))
                    return
            time.sleep(delay)


def _status_code(exc: Exception) -> int | None:
    return getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)


def _is_retryable(exc: Exception) -> bool:
    status = _status_code(exc)
    if status is not None:
        return int(status) in TRANSIENT_STATUS
    name = type(exc).__name__.lower()
    return any(part in name for part in ("timeout", "connection", "network")) or isinstance(exc, TimeoutError)


def _retry_after(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
    value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            when = parsedate_to_datetime(str(value))
            return max(0.0, (when - datetime.now(when.tzinfo or timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _language(text: str) -> str:
    if not text.strip():
        return "empty"
    DetectorFactory.seed = 2026
    try:
        return detect(text)
    except LangDetectException:
        return "undetermined"


def _response_quality_flags(text: str, prompt_text: str, finish_reason: str | None) -> list[str]:
    flags = []
    if not text.strip():
        flags.append("empty_response")
    if _language(text) not in {"en", "empty"}:
        flags.append("non_english_response")
    if "```" in text:
        flags.append("code_fence")
    if re.search(r"https?://\S+|www\.\S+", text, re.I):
        flags.append("url")
    if re.search(r"(?m)^\s{0,3}(#{1,6}\s|[-*+]\s|\d+[.)]\s)", text):
        flags.append("markdown_structure")
    if re.search(r"<\|(?:im_start|im_sep|assistant|user)|\{\s*['\"]role['\"]\s*:", text, re.I):
        flags.append("serialized_conversation_like")
    normalized_text = re.sub(r"\s+", " ", text).strip().casefold()
    normalized_prompt = re.sub(r"\s+", " ", prompt_text).strip().casefold()
    if normalized_prompt and normalized_text.startswith(normalized_prompt):
        flags.append("prompt_echo_like")
    if finish_reason in {"length", "max_tokens"}:
        flags.append("possible_truncation")
    return flags


def _safe_response_metadata(parsed: dict[str, Any]) -> dict[str, Any]:
    raw = dict(parsed.get("raw_response_metadata") or {})
    # API response objects do not normally contain headers, but explicitly drop
    # fields that could contain credentials if a proxy adds them.
    for key in list(raw):
        if ("header" in key.lower() or "authorization" in key.lower() or
                "secret" in key.lower() or "api_key" in key.lower() or "token" == key.lower()):
            raw.pop(key, None)
    return raw


def _manifest_path(run_dir: Path) -> Path:
    return run_dir / "run_manifest.json"


def _write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    temp = run_dir / "run_manifest.json.tmp"
    temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, _manifest_path(run_dir))


def _index_run(run_id: str, run_dir: Path) -> None:
    index_path = ARTIFACT_ROOT / "run_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    entries = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    entries[run_id] = str(run_dir.resolve())
    temp = index_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, index_path)


def _version_run_dirs() -> set[Path]:
    dirs = {path.parent for path in ARTIFACT_ROOT.glob("*/run_manifest.json")}
    index_path = ARTIFACT_ROOT / "run_index.json"
    if index_path.exists():
        dirs.update(Path(value) for value in json.loads(index_path.read_text(encoding="utf-8")).values())
    return {path for path in dirs if _manifest_path(path).exists()}


def _version_completed(version: str) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    for run_dir in _version_run_dirs():
        manifest = json.loads(_manifest_path(run_dir).read_text(encoding="utf-8"))
        if manifest.get("dataset_version") != version:
            continue
        for row in read_jsonl(run_dir / "responses.jsonl"):
            completed.setdefault(f"{row['prompt_id']}\0{row['model']}", row)
    return completed


def _version_attempt_counts(version: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for run_dir in _version_run_dirs():
        manifest = json.loads(_manifest_path(run_dir).read_text(encoding="utf-8"))
        if manifest.get("dataset_version") != version:
            continue
        for row in read_jsonl(run_dir / "attempts.jsonl") + read_jsonl(run_dir / "budget_events.jsonl"):
            pair = f"{row.get('prompt_id')}\0{row.get('model')}"
            counts[pair] = max(counts.get(pair, 0), int(row.get("attempt_number", 0)))
    return counts


def _assert_version_compatible(version: str, prompt_hash: str, config_hash: str) -> None:
    manifests = set(ARTIFACT_ROOT.glob("*/run_manifest.json"))
    index_path = ARTIFACT_ROOT / "run_index.json"
    if index_path.exists():
        for run_dir in json.loads(index_path.read_text(encoding="utf-8")).values():
            path = Path(run_dir) / "run_manifest.json"
            if path.exists():
                manifests.add(path)
    for path in manifests:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("dataset_version") != version:
            continue
        if existing.get("prompt_library_sha256") != prompt_hash:
            raise ConfigurationError(f"prompt library changed for {version}; use a new dataset version")
        if existing.get("configuration_sha256") != config_hash:
            raise ConfigurationError(f"generation settings changed for {version}; use a new dataset version")


def _effective_config(config: dict[str, Any], backend: str, model_names: list[str]) -> dict[str, Any]:
    if backend not in {"local", "vercel"}:
        raise ConfigurationError("backend must be local or vercel")
    models = {}
    if backend == "local":
        local = config.get("local_test_model", {})
        if not local.get("model_id"):
            raise ConfigurationError("local_test_model.model_id must be configured")
        for name in model_names:
            if name != local.get("name", "local-qwen"):
                raise ConfigurationError(f"local backend supports only {local.get('name', 'local-qwen')}")
            models[name] = dict(local, label=name)
    else:
        for name in model_names:
            if name not in config["models"]:
                raise ConfigurationError(f"unknown canonical model label: {name}")
            if not config["models"][name].get("enabled", False):
                raise ConfigurationError(f"model {name} is disabled in configuration")
            models[name] = dict(config["models"][name], label=name)
    return models


def _client_config(config: dict[str, Any], backend: str) -> tuple[str, str, int]:
    backend_cfg = config["backends"][backend]
    key_name = backend_cfg["api_key_env"]
    key = os.environ.get(key_name)
    if not key:
        raise ConfigurationError(f"environment variable {key_name} is required for {backend}")
    return backend_cfg["base_url"], key, int(config.get("request_timeout_seconds", 120))


def _config_fingerprint(config: dict[str, Any], backend: str, models: dict[str, dict[str, Any]]) -> str:
    backend_config = dict(config["backends"][backend])
    registry = config["models"] if backend == "vercel" else config.get("local_test_model", {})
    identity = {
        "dataset_version": config["dataset_version"], "backend": backend,
        "backend_config": backend_config, "model_registry": registry,
        "system_prompt": config.get("system_prompt", ""),
        "developer_prompt": config.get("developer_prompt", ""),
        "reasoning_seed": config.get("reasoning_seed", 2026),
        "reasoning_distribution": config.get("reasoning_distribution", {}),
        "max_response_characters": config.get("max_response_characters", 2000),
        "max_completion_tokens": ({k: v.get("max_completion_tokens") for k, v in registry.items()}
                                  if backend == "vercel" else registry.get("max_completion_tokens")),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _load_manifest(run_id: str) -> tuple[Path, dict[str, Any]]:
    candidate = Path(run_id)
    run_dir = candidate if candidate.is_dir() else ARTIFACT_ROOT / run_id
    path = _manifest_path(run_dir)
    if not path.exists():
        index_path = ARTIFACT_ROOT / "run_index.json"
        if index_path.exists():
            indexed = json.loads(index_path.read_text(encoding="utf-8")).get(run_id)
            if indexed:
                run_dir = Path(indexed)
                path = _manifest_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"unknown run_id {run_id!r}")
    return run_dir, json.loads(path.read_text(encoding="utf-8"))


def _journal_completed(journal: Path) -> dict[str, dict[str, Any]]:
    completed = {}
    for row in read_jsonl(journal):
        key = f"{row['prompt_id']}\0{row['model']}"
        if key in completed and completed[key].get("id") != row.get("id"):
            raise ValueError(f"duplicate successful response for {key}")
        completed[key] = row
    return completed


def _append_attempt(run_dir: Path, row: dict[str, Any]) -> None:
    _append_jsonl(run_dir / "attempts.jsonl", row)


def _append_budget_event(run_dir: Path, row: dict[str, Any]) -> None:
    _append_jsonl(run_dir / "budget_events.jsonl", row)


def _call_one(client: OpenAI, prompt: dict[str, Any], label: str, model: dict[str, Any],
              mode: str, config: dict[str, Any], run_id: str, run_dir: Path,
              limiter: RateLimiter, budget: dict[str, Any], first_attempt: int = 1) -> dict[str, Any] | None:
    prompt_text = prompt["prompt_text"]
    max_attempts = int(config.get("retry", {}).get("max_attempts", 5))
    reservation = request_cost_bound(model, prompt_text, config.get("system_prompt", ""),
        config.get("developer_prompt", "")) if config.get("budget", {}).get("enforce", False) else 0.0
    params = request_parameters(model, mode)
    messages = []
    if config.get("system_prompt"):
        messages.append({"role": "system", "content": config["system_prompt"]})
    if config.get("developer_prompt"):
        messages.append({"role": "developer", "content": config["developer_prompt"]})
    messages.append({"role": "user", "content": prompt_text})
    for attempt in range(first_attempt, max_attempts + 1):
        with _budget_lock:
            if config.get("budget", {}).get("enforce", False):
                limit = float(config["budget"]["max_spend_usd"])
                if budget["spent"] + budget["reserved"] + reservation > limit:
                    raise BudgetExhausted
                budget["reserved"] += reservation
        input_estimate = _prompt_tokens_upper_bound("\n".join(value for value in
            (config.get("system_prompt", ""), config.get("developer_prompt", ""), prompt_text) if value))
        limiter.wait(input_estimate + int(model["max_completion_tokens"]))
        start = time.monotonic()
        _append_budget_event(run_dir, {"timestamp": _utcnow(), "run_id": run_id,
            "prompt_id": prompt["prompt_id"], "model": label, "backend": config["_backend"],
            "attempt_number": attempt, "event": "reserved", "reserved_cost_usd": reservation,
            "charged_cost_usd": reservation})
        try:
            token_parameter = model.get("completion_token_parameter", "max_tokens")
            request = {"model": model["model_id"], "messages": messages,
                       token_parameter: int(model["max_completion_tokens"]), **params}
            response = client.chat.completions.create(**request)
            latency = (time.monotonic() - start) * 1000
            parsed = parse_completion(response)
            returned_model = parsed.get("returned_model_id")
            allowed = set(model.get("returned_model_ids", [])) | {model["model_id"]}
            if returned_model and returned_model not in allowed:
                raise ConfigurationError(
                    f"model identity mismatch for {label}: requested {model['model_id']}, returned {returned_model}"
                )
            text = parsed["text"]
            char_count = len(text)
            accepted = bool(text.strip()) and char_count <= int(config.get("max_response_characters", 2000))
            language = _language(text)
            quality_flags = _response_quality_flags(text, prompt_text, parsed["finish_reason"])
            cost = _actual_cost(model, parsed) if config.get("budget", {}).get("enforce", False) else 0.0
            with _budget_lock:
                budget["reserved"] -= reservation
                budget["spent"] += cost
            _append_budget_event(run_dir, {"timestamp": _utcnow(), "run_id": run_id,
                "prompt_id": prompt["prompt_id"], "model": label, "backend": config["_backend"],
                "attempt_number": attempt, "event": "settled", "reserved_cost_usd": reservation,
                "charged_cost_usd": cost, "actual_cost_usd": cost})
            raw = {
                "id": stable_response_id(config["dataset_version"], prompt["prompt_id"], label),
                "prompt_id": prompt["prompt_id"], "prompt_text": prompt_text, "model": label,
                "provider_model_id": model["model_id"], "returned_model_id": returned_model,
                "backend": config["_backend"], "provider": model.get("provider", "local" if config["_backend"] == "local" else "vercel"),
                "text": text, "reasoning_mode": mode,
                "requested_reasoning_effort": "high" if mode == "high" and model.get("reasoning_supported") else
                    ("unsupported" if mode == "high" else "default"),
                "effective_reasoning_configuration": model.get("high_effort_parameters", {}) if mode == "high" else "provider_default",
                "request_parameters_json": json.dumps(params, sort_keys=True),
                "completion_token_parameter": token_parameter,
                "input_tokens": parsed["input_tokens"], "output_tokens": parsed["output_tokens"],
                "reasoning_tokens": parsed["reasoning_tokens"], "finish_reason": parsed["finish_reason"],
                "latency_ms": round(latency, 2), "estimated_cost_usd": cost,
                "created_at": _utcnow(), "run_id": run_id, "source": "vercel_01",
                "character_count": char_count, "language": language, "contains_code_fence": "```" in text,
                "quality_flags": quality_flags,
                "accepted": accepted, "exclusion_reason": "" if accepted else
                    ("empty_response" if not text.strip() else "over_2000_characters"),
                "raw_response_metadata": _safe_response_metadata(parsed),
                "prompt_metadata_json": json.dumps(
                    {key: value for key, value in prompt.items()
                     if key not in {"prompt_id", "prompt_text"}},
                    ensure_ascii=False, sort_keys=True),
            }
            _append_attempt(run_dir, {"timestamp": _utcnow(), "run_id": run_id,
                "prompt_id": prompt["prompt_id"], "model": label, "backend": config["_backend"],
                "reasoning_mode": mode, "attempt_number": attempt, "http_status": None,
                "error_type": None, "error_classification": None, "retry_delay": 0,
                "latency_ms": round(latency, 2), "status": "success",
                "charged_cost_usd": cost, "actual_cost_usd": cost})
            _append_jsonl(run_dir / "responses.jsonl", raw)
            return raw
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            retryable = _is_retryable(exc)
            retry_delay = _retry_after(exc)
            if retryable and retry_delay is None and attempt < max_attempts:
                retry_cfg = config.get("retry", {})
                base_delay = float(retry_cfg.get("base_delay_seconds", 1))
                max_delay = float(retry_cfg.get("max_delay_seconds", 60))
                jitter = float(retry_cfg.get("jitter_seconds", 0.5))
                retry_delay = min(max_delay, base_delay * 2 ** (attempt - 1)) + random.Random(
                    f"{run_id}:{prompt['prompt_id']}:{label}:{attempt}"
                ).uniform(0, jitter)
            _append_attempt(run_dir, {"timestamp": _utcnow(), "run_id": run_id,
                "prompt_id": prompt["prompt_id"], "model": label, "backend": config["_backend"],
                "reasoning_mode": mode, "attempt_number": attempt, "http_status": _status_code(exc),
                "error_type": type(exc).__name__,
                "error_classification": ("transient" if retryable else
                    ("configuration" if isinstance(exc, ConfigurationError) or _status_code(exc) in {400, 401, 403, 404} else "permanent")),
                "retry_delay": retry_delay or 0,
                "latency_ms": round(latency, 2), "status": "retry" if retryable and attempt < max_attempts else "failed",
                "charged_cost_usd": reservation if _status_code(exc) not in {400, 401, 403, 404, 429} else 0.0})
            charged = reservation if _status_code(exc) not in {400, 401, 403, 404, 429} else 0.0
            _append_budget_event(run_dir, {"timestamp": _utcnow(), "run_id": run_id,
                "prompt_id": prompt["prompt_id"], "model": label, "backend": config["_backend"],
                "attempt_number": attempt, "event": "settled", "reserved_cost_usd": reservation,
                "charged_cost_usd": charged})
            with _budget_lock:
                budget["reserved"] = max(0.0, budget["reserved"] - reservation)
                # For ambiguous transport failures, reserve a conservative charge;
                # 429 responses are known not to have completed a generation.
                if config.get("budget", {}).get("enforce", False):
                    budget["spent"] += charged
            if isinstance(exc, ConfigurationError) or not retryable or attempt >= max_attempts:
                if _status_code(exc) in {400, 401, 403, 404}:
                    raise ConfigurationError(f"non-retryable API/configuration error ({_status_code(exc)}): {exc}") from exc
                raise
            time.sleep(float(retry_delay or 0))
    return None


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_parquet(tmp_name, index=False)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def export_run(run_id: str, output_dir: str | Path | None = None) -> dict[str, Any]:
    run_dir, manifest = _load_manifest(run_id)
    output_dir = Path(output_dir) if output_dir else PROCESSED
    all_rows = []
    all_budget_events = []
    # A dataset version can span multiple runs; aggregate successful journals and
    # reject conflicting stable IDs rather than silently overwriting provenance.
    manifests = {_manifest_path(path) for path in _version_run_dirs()}
    manifests.add(_manifest_path(run_dir))
    for candidate in sorted(manifests):
        other = json.loads(candidate.read_text(encoding="utf-8"))
        if other.get("dataset_version") == manifest["dataset_version"]:
            all_rows.extend(read_jsonl(candidate.parent / "responses.jsonl"))
            all_budget_events.extend(read_jsonl(candidate.parent / "budget_events.jsonl"))
    by_id: dict[str, dict[str, Any]] = {}
    for row in all_rows:
        old = by_id.get(row["id"])
        if old and (old.get("text"), old.get("prompt_text"), old.get("model")) != (
                row.get("text"), row.get("prompt_text"), row.get("model")):
            raise ValueError(f"conflicting journal records for stable response ID {row['id']}; use a new dataset version")
        by_id[row["id"]] = row
    rows = sorted(by_id.values(), key=lambda row: (row["prompt_id"], row["model"], row["id"]))
    latest_budget_events: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for event in all_budget_events:
        key = (event.get("run_id", ""), event.get("prompt_id", ""), event.get("model", ""),
               int(event.get("attempt_number", 0)))
        latest_budget_events[key] = event
    charged_by_model: dict[str, float] = {}
    for event in latest_budget_events.values():
        label = str(event.get("model", "unknown"))
        charged_by_model[label] = charged_by_model.get(label, 0.0) + float(event.get("charged_cost_usd", 0) or 0)
    accepted = [row for row in rows if row.get("accepted") and row.get("text", "").strip()]
    dataset = pd.DataFrame(accepted)
    if len(dataset) and "prompt_metadata_json" in dataset:
        metadata = dataset.prompt_metadata_json.fillna("{}").map(json.loads)
        optional = pd.json_normalize(metadata).add_prefix("prompt_")
        dataset = pd.concat([dataset.reset_index(drop=True), optional.reset_index(drop=True)], axis=1)
    required = ["id", "prompt_id", "text", "model"]
    if dataset.empty:
        dataset = pd.DataFrame(columns=required + ["source"])
    for col in required:
        if col not in dataset:
            dataset[col] = pd.Series(dtype="object")
    version = str(manifest["dataset_version"])
    artifact = output_dir / f"dataset_{version}.parquet"
    profile_path = output_dir / f"dataset_profile_{version}.json"
    _atomic_parquet(dataset, artifact)
    char_lengths = dataset.text.fillna("").astype(str).str.len() if len(dataset) else pd.Series(dtype=float)
    word_lengths = dataset.text.fillna("").astype(str).str.split().str.len() if len(dataset) else pd.Series(dtype=float)
    model_counts = dataset.model.value_counts().sort_index().to_dict() if len(dataset) else {}
    mode_counts = dataset.reasoning_mode.value_counts().sort_index().to_dict() if len(dataset) and "reasoning_mode" in dataset else {}
    rejected = [row for row in rows if not row.get("accepted")]
    group_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        key = f"{row.get('model')}|{row.get('reasoning_mode')}"
        target = "accepted" if row.get("accepted") else "rejected"
        group_counts.setdefault(key, {"accepted": 0, "rejected": 0})[target] += 1
    profile = {
        "dataset_version": manifest["dataset_version"], "dataset_sha256": sha256_file(artifact),
        "prompt_library_sha256": manifest["prompt_library_sha256"],
        "configuration_sha256": manifest["configuration_sha256"], "rows": int(len(dataset)),
        "generated_rows": len(rows), "accepted_rows": len(accepted),
        "rejected_rows": len(rows) - len(accepted),
        "rejected_by_reason": pd.Series([r.get("exclusion_reason", "") for r in rejected]).value_counts().to_dict(),
        "accepted_rejected_by_model_reasoning_mode": group_counts,
        "class_counts": {str(k): int(v) for k, v in model_counts.items()},
        "classes": sorted(dataset.model.astype(str).unique().tolist()) if len(dataset) else [],
        "reasoning_mode_counts": {str(k): int(v) for k, v in mode_counts.items()},
        "character_length": {"p10": float(char_lengths.quantile(.1)) if len(char_lengths) else None,
            "median": float(char_lengths.median()) if len(char_lengths) else None,
            "p90": float(char_lengths.quantile(.9)) if len(char_lengths) else None},
        "word_length": {"p10": float(word_lengths.quantile(.1)) if len(word_lengths) else None,
            "median": float(word_lengths.median()) if len(word_lengths) else None,
            "p90": float(word_lengths.quantile(.9)) if len(word_lengths) else None},
        "input_tokens": int(sum(int(r.get("input_tokens", 0)) for r in rows)),
        "output_tokens": int(sum(int(r.get("output_tokens", 0)) for r in rows)),
        "reasoning_tokens": int(sum(int(r.get("reasoning_tokens", 0)) for r in rows)),
        "estimated_cost_usd": float(sum(charged_by_model.values()) if all_budget_events else
                                     sum(float(r.get("estimated_cost_usd", 0)) for r in rows)),
        "estimated_cost_usd_by_model": charged_by_model,
        "language_counts": pd.Series([str(r.get("language", "unknown")) for r in rows]).value_counts().to_dict(),
        "code_fence_rows": int(sum(bool(r.get("contains_code_fence")) for r in rows)),
        "quality_flag_counts": pd.Series([flag for row in rows for flag in row.get("quality_flags", [])]).value_counts().to_dict(),
        "finish_reason_counts": pd.Series([str(r.get("finish_reason", "unknown")) for r in rows]).value_counts().to_dict(),
        "provider_model_ids": sorted({str(r.get("provider_model_id")) for r in rows}),
        "model_routes": sorted({json.dumps({"backend": r.get("backend"), "provider": r.get("provider"),
            "provider_model_id": r.get("provider_model_id"), "returned_model_id": r.get("returned_model_id")},
            sort_keys=True) for r in rows}),
        "model_statistics": {},
        "source": "vercel_01 append-only response journals",
    }
    for row in rows:
        stats = profile["model_statistics"].setdefault(row["model"], {
            "generated": 0, "accepted": 0, "rejected": 0, "input_tokens": 0,
            "output_tokens": 0, "reasoning_tokens": 0, "estimated_cost_usd": 0.0})
        stats["generated"] += 1
        stats["accepted" if row.get("accepted") else "rejected"] += 1
        for field in ("input_tokens", "output_tokens", "reasoning_tokens"):
            stats[field] += int(row.get(field, 0))
    for label, charged in charged_by_model.items():
        profile["model_statistics"].setdefault(label, {
            "generated": 0, "accepted": 0, "rejected": 0, "input_tokens": 0,
            "output_tokens": 0, "reasoning_tokens": 0, "estimated_cost_usd": 0.0
        })["estimated_cost_usd"] = charged
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rejected_path = run_dir / "rejected_responses.csv"
    run_rows = read_jsonl(run_dir / "responses.jsonl")
    pd.DataFrame([r for r in run_rows if not r.get("accepted")]).to_csv(rejected_path, index=False)
    _maybe_register_dataset(artifact, profile_path, profile)
    return {"artifact": str(artifact), "profile": str(profile_path), **profile}


def _maybe_register_dataset(artifact: Path, profile_path: Path, profile: dict[str, Any]) -> bool:
    if set(profile.get("classes", [])) != set(LABELS):
        return False
    frame = pd.read_parquet(artifact)
    if not {"id", "prompt_id", "text", "model"}.issubset(frame.columns):
        return False
    if (frame.id.isna().any() or frame.id.astype(str).duplicated().any() or
            frame.text.fillna("").astype(str).str.strip().eq("").any()):
        return False
    if set(frame.model.astype(str)) != set(LABELS):
        return False
    # Register only a complete, validated 18-class artifact. Preserve the catalog
    # as found and add one entry; docs are rendered from this source of truth.
    from .dataset_catalog import DEFAULT_CATALOG, DEFAULT_MARKDOWN, load_dataset_catalog, render_markdown
    catalog = load_dataset_catalog(DEFAULT_CATALOG)
    dataset_id = str(profile.get("dataset_version", "vercel_v01"))
    if dataset_id == "vercel_local_test":
        return False
    if any(entry["id"] == dataset_id for entry in catalog["datasets"]):
        return True
    root = ROOT
    try:
        relative_artifact = artifact.resolve().relative_to(root.resolve()).as_posix()
        relative_profile = profile_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return False
    entry = {
        "id": dataset_id, "name": f"Vercel {dataset_id.removeprefix('vercel_')} — Generated Responses",
        "section": "Generated datasets",
        "description": "Responses generated from the independently managed English prompt library using the configured Vercel AI Gateway models.",
        "artifact": relative_artifact, "profile": relative_profile,
        "validation": "Generated text dataset; use the frozen V0 validation set for classifier experiments.",
    }
    # Append only the new list entry so hand-authored catalog formatting and
    # comments for every existing dataset are left intact.
    entry_yaml = yaml.safe_dump([entry], sort_keys=False, allow_unicode=True, width=100)
    addition = "\n" + "\n".join("  " + line if line else line for line in entry_yaml.rstrip().splitlines()) + "\n"
    with DEFAULT_CATALOG.open("a", encoding="utf-8") as stream:
        stream.write(addition)
    catalog = load_dataset_catalog(DEFAULT_CATALOG)
    DEFAULT_MARKDOWN.write_text(render_markdown(catalog, ROOT), encoding="utf-8")
    return True


def _stats(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    rows = read_jsonl(run_dir / "responses.jsonl")
    attempts = read_jsonl(run_dir / "attempts.jsonl")
    budget_events = read_jsonl(run_dir / "budget_events.jsonl")
    completed = {(row["prompt_id"], row["model"]) for row in rows}
    failed = {(row["prompt_id"], row["model"]) for row in attempts if row.get("status") == "failed"}
    latest_attempt: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in budget_events:
        key = (row.get("prompt_id", ""), row.get("model", ""), int(row.get("attempt_number", 0)))
        latest_attempt[key] = row
    if budget_events:
        estimated_spend = sum(float(row.get("charged_cost_usd", row.get("reserved_cost_usd", 0)) or 0)
                              for row in latest_attempt.values())
    else:
        estimated_spend = sum(float(row.get("charged_cost_usd", 0) or 0) for row in attempts)
    by_model: dict[str, dict[str, Any]] = {}
    for row in rows:
        stats = by_model.setdefault(row["model"], {"completed": 0, "accepted": 0, "rejected": 0,
            "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "estimated_cost_usd": 0.0})
        stats["completed"] += 1
        stats["accepted" if row.get("accepted") else "rejected"] += 1
        for field in ("input_tokens", "output_tokens", "reasoning_tokens"):
            stats[field] += int(row.get(field, 0))
    for event in latest_attempt.values():
        label = str(event.get("model", "unknown"))
        stats = by_model.setdefault(label, {"completed": 0, "accepted": 0, "rejected": 0,
            "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "estimated_cost_usd": 0.0})
        stats["estimated_cost_usd"] += float(event.get("charged_cost_usd", 0) or 0)
    failed_pairs = len(failed - completed)
    expected = int(manifest.get("expected_pairs", 0))
    return {"completed": len(completed), "failed_pairs": failed_pairs,
        "remaining": max(0, expected - len(completed) - failed_pairs),
        "accepted": sum(bool(row.get("accepted")) for row in rows),
        "rejected": sum(not bool(row.get("accepted")) for row in rows),
        "attempts": len(attempts), "retries": sum(row.get("status") == "retry" for row in attempts),
        "input_tokens": sum(int(r.get("input_tokens", 0)) for r in rows),
        "output_tokens": sum(int(r.get("output_tokens", 0)) for r in rows),
        "reasoning_tokens": sum(int(r.get("reasoning_tokens", 0)) for r in rows),
        "estimated_cost_usd": estimated_spend, "per_model": by_model,
        "expected": expected}


def plan(prompts_path: str | Path, backend: str, model_names: list[str] | None = None,
         config_path: str | Path = DEFAULT_CONFIG, limit: int | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    if backend == "local":
        config["dataset_version"] = str(config.get("local_test_model", {}).get("dataset_version", "vercel_local_test"))
    prompts, prompt_hash, violations = load_prompts(prompts_path)
    all_eligible = prompts[~prompts.prompt_id.isin(violations.prompt_id)].copy()
    mode_map = assign_reasoning_modes(all_eligible.prompt_id.astype(str),
        int(config.get("reasoning_seed", 2026)), float(config["reasoning_distribution"]["high"]))
    if limit is not None:
        prompts = prompts.head(limit)
    model_names = model_names or (["local-qwen"] if backend == "local" else list(LABELS))
    models = _effective_config(config, backend, model_names)
    eligible = prompts[~prompts.prompt_id.isin(violations.prompt_id)].copy()
    _validate_tpm_limits(models, eligible, config["throttling"].get("tokens_per_minute"),
                         config.get("system_prompt", ""), config.get("developer_prompt", ""))
    mode_counts = {"default": 0, "high": 0, "unsupported": 0}
    total_cost = 0.0
    by_model = {}
    for label, model in models.items():
        calls, cost = 0, 0.0
        for row in eligible.itertuples(index=False):
            mode = mode_map[row.prompt_id]
            if mode == "high" and not model.get("reasoning_supported"):
                mode_counts["unsupported"] += 1
            else:
                mode_counts[mode] += 1
            if backend == "vercel":
                cost += request_cost_bound(model, row.prompt_text, config.get("system_prompt", ""),
                                           config.get("developer_prompt", ""))
            calls += 1
        by_model[label] = {"calls": calls, "max_reserved_cost_usd": cost,
                           "requests_per_minute": model.get("requests_per_minute", config["throttling"]["requests_per_minute"])}
        total_cost += cost
    result = {"prompts": len(prompts), "eligible_prompts": len(eligible),
        "prompt_library_sha256": prompt_hash, "prompt_violations": len(violations),
        "prompt_violations_by_flag": ({str(k): int(v) for k, v in
            violations["flags"].str.split(";").explode().value_counts().items()} if len(violations) else {}),
        "prompt_violation_id_sample": violations.prompt_id.head(20).tolist(),
        "total_calls": sum(row["calls"] for row in by_model.values()),
        "reasoning_mode_counts": mode_counts, "models": by_model,
        "estimated_max_spend_usd": total_cost,
        "budget_usd": config["budget"]["max_spend_usd"] if backend == "vercel" else None,
        "fits_budget": backend != "vercel" or total_cost <= config["budget"]["max_spend_usd"]}
    print(json.dumps(result, indent=2))
    return result


def generate(prompts_path: str | Path | None = None, *, backend: str | None = None,
             model_names: list[str] | None = None, all_models: bool = False,
             config_path: str | Path = DEFAULT_CONFIG, limit: int | None = None,
             output_dir: str | Path | None = None, resume: str | None = None,
             budget_override: float | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    if resume:
        run_dir, manifest = _load_manifest(resume)
        saved_config_path = Path(manifest.get("configuration_file", config_path))
        if (Path(config_path).resolve() == DEFAULT_CONFIG.resolve() and
                saved_config_path.is_file() and saved_config_path.resolve() != DEFAULT_CONFIG.resolve()):
            config_path = saved_config_path
            config = load_config(config_path)
        backend = manifest["backend"]
        prompts_path = manifest["prompt_path"]
        if sha256_file(prompts_path) != manifest["prompt_library_sha256"]:
            raise ConfigurationError("prompt CSV hash changed; resume refused")
        model_names = manifest["models"]
        limit = manifest.get("limit")
    else:
        if not prompts_path or not backend:
            raise ValueError("prompts_path and backend are required for a new run")
        if all_models:
            model_names = list(LABELS)
        model_names = model_names or (["local-qwen"] if backend == "local" else [])
        if not model_names:
            raise ValueError("select --model/--models or --all-models")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        run_dir = Path(output_dir) if output_dir else ARTIFACT_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        prompts_path = str(Path(prompts_path).resolve())
    if backend == "local":
        config["dataset_version"] = str(config.get("local_test_model", {}).get("dataset_version", "vercel_local_test"))
    prompts, prompt_hash, violations = load_prompts(prompts_path)
    all_eligible = prompts[~prompts.prompt_id.isin(violations.prompt_id)].copy()
    mode_map = assign_reasoning_modes(all_eligible.prompt_id.astype(str),
        int(config.get("reasoning_seed", 2026)), float(config["reasoning_distribution"]["high"]))
    if limit is not None:
        prompts = prompts.head(limit)
    models = _effective_config(config, backend, list(model_names))
    _validate_tpm_limits(models, all_eligible, config["throttling"].get("tokens_per_minute"),
                         config.get("system_prompt", ""), config.get("developer_prompt", ""))
    if backend == "vercel":
        for model in models.values():
            request_cost_bound(model, "")  # fail before any call if price data is missing
    fingerprint = _config_fingerprint(config, backend, models)
    if resume:
        if fingerprint != manifest.get("configuration_sha256"):
            raise ConfigurationError("generation configuration changed; create a new dataset version/run")
        if prompt_hash != manifest.get("prompt_library_sha256"):
            raise ConfigurationError("prompt library hash changed; resume refused")
        run_id = resume
        if manifest.get("status") in {"completed", "budget_exhausted"}:
            # Completed runs remain exportable; budget-exhausted runs may resume
            # only after an explicitly larger budget override.
            if manifest.get("status") == "completed":
                return export_run(run_id)
        manifest["status"] = "running"
    else:
        _assert_version_compatible(config["dataset_version"], prompt_hash, fingerprint)
        run_id = run_dir.name
        eligible_prompt_ids = set(prompts.loc[~prompts.prompt_id.isin(violations.prompt_id), "prompt_id"].astype(str))
        prior_pairs = _version_completed(config["dataset_version"])
        prior_selected = sum(key.split("\0", 1)[0] in eligible_prompt_ids and
                             key.split("\0", 1)[1] in models for key in prior_pairs)
        planned_pairs = len(eligible_prompt_ids) * len(models)
        manifest = {"run_id": run_id, "dataset_version": config["dataset_version"],
            "backend": backend, "prompt_path": str(Path(prompts_path).resolve()),
            "prompt_library_sha256": prompt_hash, "configuration_sha256": fingerprint,
            "configuration_file": str(Path(config_path).resolve()),
            "configuration_file_sha256": sha256_file(config_path),
            "configuration": {"backend_base_url": config["backends"][backend]["base_url"],
                "system_prompt": config.get("system_prompt", ""),
                "developer_prompt": config.get("developer_prompt", ""),
                "reasoning_seed": config.get("reasoning_seed", 2026),
                "reasoning_distribution": config.get("reasoning_distribution", {}),
                "max_response_characters": config.get("max_response_characters", 2000),
                "models": {name: {"model_id": value["model_id"],
                    "request_parameters": value.get("request_parameters", {}),
                    "high_effort_parameters": value.get("high_effort_parameters", {}),
                    "max_completion_tokens": value.get("max_completion_tokens"),
                    "pricing": value.get("pricing", {})} for name, value in models.items()},
                "throttling": config.get("throttling", {}), "retry": config.get("retry", {}),
                "budget": config.get("budget", {})},
            "models": list(models), "model_ids": {k: v["model_id"] for k, v in models.items()},
            "limit": limit, "expected_pairs": int(planned_pairs - prior_selected),
            "already_completed_pairs": int(prior_selected),
            "prompt_rows": int(len(prompts)),
            "prompt_violations": len(violations), "prompt_violation_ids": violations.prompt_id.tolist(),
            "started_at": _utcnow(), "status": "running", "budget_limit_usd": budget_override or config["budget"]["max_spend_usd"]}
        _write_manifest(run_dir, manifest)
        _index_run(run_id, run_dir)
    if not resume:
        if not violations.empty:
            violations.to_csv(run_dir / "prompt_violations.csv", index=False)
    else:
        manifest = json.loads(_manifest_path(run_dir).read_text(encoding="utf-8"))
        manifest["status"] = "running"
    config = dict(config)
    config["_backend"] = backend
    # Production spending controls must never block local vLLM smoke tests.
    if backend == "local":
        config["budget"] = dict(config["budget"], enforce=False)
    if budget_override is not None:
        config["budget"] = dict(config["budget"], max_spend_usd=float(budget_override))
        manifest["budget_limit_usd"] = float(budget_override)
    base_url, api_key, timeout = _client_config(config, backend)
    # No HTTP calls occur until create() is invoked by a selected prompt worker.
    limiter_by_model = {label: RateLimiter(float(model.get("requests_per_minute", config["throttling"]["requests_per_minute"])),
        model.get("tokens_per_minute", config["throttling"].get("tokens_per_minute")))
                        for label, model in models.items()}
    budget = {"spent": _stats(run_dir, manifest)["estimated_cost_usd"], "reserved": 0.0}
    completed = _version_completed(config["dataset_version"])
    completed.update(_journal_completed(run_dir / "responses.jsonl"))
    attempt_counts = _version_attempt_counts(config["dataset_version"])
    for attempt_row in read_jsonl(run_dir / "attempts.jsonl"):
        pair = f"{attempt_row['prompt_id']}\0{attempt_row['model']}"
        attempt_counts[pair] = max(attempt_counts.get(pair, 0), int(attempt_row.get("attempt_number", 0)))
    eligible = prompts[~prompts.prompt_id.isin(violations.prompt_id)]
    eligible_rows = eligible.to_dict(orient="records")
    errors = []
    budget_stop = False
    concurrency = max(1, int(config["throttling"].get("concurrency", 1)))

    def worker(task: tuple[dict[str, Any], str, dict[str, Any], str, int]) -> None:
        row, label, model, mode, first_attempt = task
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        _call_one(client, row, label, model, mode, config, run_id, run_dir,
                  limiter_by_model[label], budget, first_attempt=first_attempt)

    try:
        # Finish one model before starting the next by default, matching the
        # requested low-concurrency operating mode and simplifying cost control.
        for label, model in models.items():
            tasks = []
            for row in eligible_rows:
                pair = f"{row['prompt_id']}\0{label}"
                if pair not in completed and attempt_counts.get(pair, 0) < int(config.get("retry", {}).get("max_attempts", 5)):
                    tasks.append((row, label, model, mode_map[row["prompt_id"]],
                                  attempt_counts.get(pair, 0) + 1))
            done = 0
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(worker, task) for task in tasks]
                for future in as_completed(futures):
                    done += 1
                    if future.cancelled():
                        continue
                    try:
                        future.result()
                    except BudgetExhausted:
                        for item in futures:
                            item.cancel()
                        manifest["status"] = "budget_exhausted"
                        budget_stop = True
                    except Exception as exc:
                        errors.append(f"{label}: {type(exc).__name__}: {exc}")
                    if done % 25 == 0 or done == len(futures):
                        live = _stats(run_dir, manifest)
                        print(f"{label}: {done}/{len(futures)} attempted; "
                              f"total complete {live['completed']}/{live['expected']}; "
                              f"accepted {live['accepted']}, rejected {live['rejected']}; "
                              f"estimated spend ${budget['spent']:.4f}")
                    if budget_stop:
                        break
            if budget_stop:
                break
        result = export_run(run_id, output_dir=PROCESSED)
        manifest["updated_at"] = _utcnow()
        manifest["statistics"] = _stats(run_dir, manifest)
        manifest["errors"] = errors
        if manifest.get("status") != "budget_exhausted":
            manifest["status"] = "failed" if errors or manifest["statistics"]["failed_pairs"] or manifest["statistics"]["remaining"] else "completed"
        manifest["budget_spent_estimated_usd"] = budget["spent"]
        _write_manifest(run_dir, manifest)
        print(json.dumps({"run_id": run_id, "status": manifest["status"],
                          "statistics": manifest["statistics"], "budget_spent_estimated_usd": budget["spent"],
                          "budget_remaining_usd": max(0, manifest["budget_limit_usd"] - budget["spent"]),
                          "export": result["artifact"], "errors": errors}, indent=2))
        return {"run_id": run_id, "status": manifest["status"], "statistics": manifest["statistics"],
                "artifact": result["artifact"], "errors": errors}
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["updated_at"] = _utcnow()
        manifest["statistics"] = _stats(run_dir, manifest)
        _write_manifest(run_dir, manifest)
        export_run(run_id)
        raise


def status(run_id: str) -> dict[str, Any]:
    run_dir, manifest = _load_manifest(run_id)
    stats = _stats(run_dir, manifest)
    result = {"run_id": run_id, "status": manifest.get("status"), "statistics": stats,
              "budget_spent_estimated_usd": stats["estimated_cost_usd"],
              "budget_remaining_usd": max(0.0, float(manifest.get("budget_limit_usd", 0)) - stats["estimated_cost_usd"]),
              "models": manifest.get("models"), "started_at": manifest.get("started_at")}
    print(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--prompts", required=True, type=Path)
    p.add_argument("--backend", choices=["local", "vercel"], required=True)
    p.add_argument("--model", action="append")
    p.add_argument("--models")
    p.add_argument("--all-models", action="store_true")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--limit", type=int)
    g = sub.add_parser("generate")
    g.add_argument("--prompts", type=Path)
    g.add_argument("--backend", choices=["local", "vercel"])
    g.add_argument("--model", action="append")
    g.add_argument("--models")
    g.add_argument("--all-models", action="store_true")
    g.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    g.add_argument("--limit", type=int)
    g.add_argument("--output-dir", type=Path)
    g.add_argument("--resume")
    g.add_argument("--budget-override", type=float)
    s = sub.add_parser("status")
    s.add_argument("--run-id", required=True)
    e = sub.add_parser("export")
    e.add_argument("--run-id", required=True)
    e.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    model_names = getattr(args, "model", None) or []
    if getattr(args, "models", None):
        model_names += [x.strip() for x in args.models.split(",") if x.strip()]
    if getattr(args, "all_models", False):
        model_names = list(LABELS)
    if args.command == "plan":
        plan(args.prompts, args.backend, model_names or None, args.config, args.limit)
    elif args.command == "generate":
        generate(args.prompts, backend=args.backend, model_names=model_names or None,
                 all_models=args.all_models, config_path=args.config, limit=args.limit,
                 output_dir=args.output_dir, resume=args.resume, budget_override=args.budget_override)
    elif args.command == "status":
        status(args.run_id)
    else:
        print(json.dumps(export_run(args.run_id, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
