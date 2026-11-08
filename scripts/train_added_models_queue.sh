#!/usr/bin/env bash
set -uo pipefail
cd /home/zodrak/nlm
mkdir -p artifacts/logs
LOG=artifacts/logs/added_models_training_queue.log
exec >>"$LOG" 2>&1
printf '\n[%s] Queue started. Waiting for current Qwen4 run and model downloads.\n' "$(date -Is)"

wait_until_inactive() {
  local unit="$1"
  while systemctl --user is-active --quiet "$unit"; do
    sleep 30
  done
  local result
  result=$(systemctl --user show "$unit" -p Result --value 2>/dev/null || true)
  printf '[%s] %s finished (result=%s).\n' "$(date -Is)" "$unit" "${result:-unknown}"
  [[ "$result" == success ]]
}

wait_until_inactive nlm-qwen4-10epoch.service
for unit in \
  nlm-llama32-3b-download.service \
  nlm-ministral3-3b-download.service \
  nlm-smollm3-3b-download.service \
  nlm-granite4-h-micro-download.service \
  nlm-embeddinggemma2-download.service \
  nlm-gemma4-e4b-download.service; do
  if ! wait_until_inactive "$unit"; then
    printf '[%s] Download service %s failed; refusing to start training before all models are local.\n' "$(date -Is)" "$unit"
    exit 1
  fi
done

failures=()
for model in qwen4 qwen17 llama32_3b embeddinggemma2 gemma4_e4b ministral3_3b smollm3_3b granite4_350m catboost_stylometry; do
  printf '\n[%s] Starting %s.\n' "$(date -Is)" "$model"
  if HF_HUB_OFFLINE=1 /home/zodrak/cuda13.x/bin/python -u -m src.pipeline models train \
      --model "$model" --dataset v0 --config "configs/models/${model}.yaml"; then
    printf '[%s] Finished %s successfully.\n' "$(date -Is)" "$model"
    /home/zodrak/cuda13.x/bin/python - "$model" <<'PY'
import json, sys
from pathlib import Path
model = sys.argv[1]
paths = list(Path("artifacts/predictions").glob(f"{model}_v0_*_manifest.json"))
if paths:
    manifest = json.loads(max(paths, key=lambda path: path.stat().st_mtime).read_text())
    epoch_metrics = manifest.get("validation_metrics_by_epoch", [])
    accuracies = [round(float(item["eval_accuracy"]), 5)
                  for item in epoch_metrics if item.get("eval_accuracy") is not None]
    print(f"SELECTED model={model} epoch={manifest.get('selected_epoch')} "
          f"iteration={manifest.get('selected_iteration')} "
          f"validation_accuracy={manifest.get('selected_validation_accuracy')} "
          f"epoch_accuracies={accuracies}", flush=True)
else:
    print(f"No experiment manifest found for {model}", flush=True)
PY
  else
    status=$?
    printf '[%s] FAILED %s (exit=%s). Continuing queue.\n' "$(date -Is)" "$model" "$status"
    failures+=("$model:$status")
  fi
done
printf '\n[%s] Queue finished. Failures: %s\n' "$(date -Is)" "${failures[*]:-none}"
