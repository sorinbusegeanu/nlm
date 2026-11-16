# Vercel v01 response generation

`src.vercel_01` generates independent responses from a supplied prompt CSV. It is
separate from classifier training. The input prompt library remains read-only;
prompt flags are written to a run audit and flagged prompts are skipped.

## Install and configure

Install project dependencies (the OpenAI-compatible Python SDK is included in
`requirements.txt`):

```bash
python -m pip install -r requirements.txt
```

The configured backend endpoints and credential variable names are in
[`configs/vercel_01.yaml`](../configs/vercel_01.yaml). Provide a key through the
environment only:

```bash
export VLLM_API_KEY=local-test-key
# or
export AI_GATEWAY_API_KEY=...
```

For local vLLM, configure `local_test_model.model_id` to the exact served ID. The
local mapping is named `local-qwen` and is not one of the 18 competition labels.
The default local model mapping is a smoke-test placeholder; change it to match
the server before use.

The production registry contains the 18 requested NLM labels and exact Gateway
IDs. Pricing values, source URLs and checked date are recorded per model in YAML.
They are estimates for enforcement; provider billing can differ due to routing,
promotions, long-context tiers, caching, and reasoning tokens. Review the linked
Gateway price pages before a production run. The configured production budget is
$300 per run and is enforced before requests using a conservative token estimate.
A run stops and exports its partial data when the remaining budget cannot reserve
the maximum configured request cost. The budget is not raised automatically.

## Prompt CSV

Required columns:

```csv
prompt_id,prompt_text
P00001,Explain why stars twinkle.
P00002,Describe the philosophical implications of free will.
```

IDs must be unique and both fields nonempty. Extra CSV columns are preserved in
response journals as `prompt_metadata_json` and flattened into the export with a
`prompt_` prefix. Prompts are evaluated with a deterministic language detector
and coding/programming heuristics. Any flagged prompt is listed in
`prompt_violations.csv` and skipped; its source CSV is never changed.

## Plan before generation

```bash
python -m src.vercel_01 plan --prompts prompts.csv --backend local --model local-qwen
python -m src.vercel_01 plan --prompts prompts.csv --backend vercel --all-models
```

The plan reports eligible prompt count, call count, the fixed 70/30 default/high
reasoning split, configured rates, maximum reserved cost and budget fit. For
models without a high-effort control, the shared high-assigned prompts use that
model's default behavior and record `unsupported`.

## Generate and resume

```bash
python -m src.vercel_01 generate --prompts prompts.csv --backend local --model local-qwen --limit 100
python -m src.vercel_01 generate --prompts prompts.csv --backend vercel --model glm-5.3
python -m src.vercel_01 generate --prompts prompts.csv --backend vercel --all-models
python -m src.vercel_01 generate --resume <run_id>
python -m src.vercel_01 status --run-id <run_id>
python -m src.vercel_01 export --run-id <run_id>
```

Each prompt gets a fresh one-message request (plus an explicitly configured
system or developer prompt, both empty by default). Prompts are sorted by `prompt_id`; the 70/30
reasoning assignment is prompt-stable across models, using seed 2026. Requests
are throttled per model and transient failures retry up to five total attempts
with exponential backoff and `Retry-After` support. Authentication, model and
parameter failures are not retried. Model identity mismatches fail the pair.

A successful response is flushed and fsynced to `responses.jsonl` immediately.
That journal is append-only and durable; interrupted runs can be resumed by
`run_id`. Resume verifies the input CSV hash and generation configuration and
skips successful pairs. Changing generation identity requires a new dataset
version; `vercel_v01` is frozen in this implementation. The stable response ID
is a SHA-256 of dataset version, prompt ID and NLM label.

Responses longer than 2,000 Python characters and empty outputs remain available
in the raw journal and rejected audit, but are omitted from the accepted Parquet
export. Other observed properties (language, code fences and finish reason) are
recorded; non-English model output is flagged rather than rejected. No response
is regenerated only because it exceeds the length limit.

## Outputs

Each run writes:

```text
artifacts/vercel_01/<run_id>/
  responses.jsonl              # durable response journal
  attempts.jsonl                # request/retry/failure events
  budget_events.jsonl           # crash-safe cost reservations and settlements
  run_manifest.json             # hashes, configuration, status and totals
  prompt_violations.csv         # flagged prompt rows, when present
  rejected_responses.csv        # excluded output rows
```

Exports are atomically rebuilt at:

```text
data/processed/dataset_vercel_v01.parquet
data/processed/dataset_profile_vercel_v01.json
```

The Parquet contains NLM's `id`, `prompt_id`, `text`, and `model`, plus generation
and prompt metadata. Export combines journals for the same version and refuses
conflicting stable IDs. The dataset catalog is updated only when all 18 labels
are present and the schema/integrity checks pass. The generation factory does
not train or evaluate classifiers.

## Operational note

Use `plan` first. Do not assume a prompt-library estimate equals provider billing:
Gateway can route across providers, input token estimates are heuristic, and
provider price schedules may change. `estimated_cost_usd` is a local estimate
using the configured per-model rates and reported token usage.
