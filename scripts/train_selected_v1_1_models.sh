#!/usr/bin/env bash
set -u

cd /home/zodrak/nlm
python=/home/zodrak/cuda13.x/bin/python
queue_dir=/home/zodrak/nlm/artifacts/runs/v1_1_selected_models_20261104
config_dir="$queue_dir/configs"
start_index=${1:-0}
# Queue plan on v1.1: train each model below in order and refresh predictions
# before moving to the next. CatBoost first builds its stylometry feature cache.
# When resuming after Gemma, refresh its saved checkpoint once before Granite;
# Gemma started before the automatic post-training prediction hook existed.
models=(gemma4_e2b granite4_350m qwen06 ministral3_3b qwen17 llama32_3b smollm3_3b catboost_stylometry ultrabert380 qwen4 phi35_mini)
steps=(
  "Train Gemma 4 E2B; refresh predictions"
  "Train Granite 4 350M; refresh predictions"
  "Train Qwen3 0.6B; refresh predictions"
  "Train Ministral 3 3B; refresh predictions"
  "Train Qwen3 1.7B; refresh predictions"
  "Train Llama 3.2 3B; refresh predictions"
  "Train SmolLM3 3B; refresh predictions"
  "Build CatBoost stylometry features; train CatBoost + Stylometry; refresh predictions"
  "Train UltraBERT-380M; refresh predictions"
  "Train Qwen3 4B; refresh predictions"
  "Train Microsoft Phi-3.5 Mini with LoRA; refresh predictions"
)
mkdir -p "$config_dir"

"$python" - "$config_dir" "${models[@]}" <<'PY'
import sys
from pathlib import Path
import yaml

out = Path(sys.argv[1])
for model in sys.argv[2:]:
    source = Path("configs/models") / f"{model}.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["warmup_epochs"] = int(config.get("epochs", 10))
    config.update({
        "epochs": 100,
        "early_stopping_patience": 3,
        "max_batch_size": 1024,
        "batch_probe_safety_fraction": 0.95,
        "learning_rate_decay": 0.8,
        "lr_scheduler_type": "epoch_decay",
        "checkpoint_selection": "highest_validation_accuracy",
        "local_files_only": True,
    })
    (out / f"{model}.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

printf '%s\n' "${models[*]}" > "$queue_dir/model_order.txt"
printf '%s QUEUE_START dataset=v1.1 start_index=%s models=%s max_epochs=100 early_stopping_patience=3 decay=0.8 offline=true\n' \
  "$(date -Is)" "$start_index" "${models[*]}" | tee -a "$queue_dir/events.log"
if (( start_index > 0 )); then
    printf '%s PLAN CATCHUP Refresh predictions for the already completed Gemma run before continuing\n' \
      "$(date -Is)" | tee -a "$queue_dir/events.log"
fi
for index in "${!steps[@]}"; do
    step_number=$((index + 1))
    (( index < start_index )) && continue
    printf '%s PLAN %02d %s\n' "$(date -Is)" "$step_number" "${steps[$index]}" \
      | tee -a "$queue_dir/events.log"
done

# A model may have completed under a trainer process started before the
# prediction-factory hook was installed. Reconcile its saved manifest before
# starting the next model in the sequence.
if (( start_index > 0 )); then
    printf '%s PREDICTION_FACTORY_CATCHUP_START before_index=%s\n' \
      "$(date -Is)" "$start_index" | tee -a "$queue_dir/events.log"
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
      TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled \
      "$python" -m src.pipeline prediction refresh --dataset v0 --data-dir data/processed \
      --artifact-dir artifacts >>"$queue_dir/events.log" 2>&1
    rc=$?
    if (( rc != 0 )); then
        printf '%s PREDICTION_FACTORY_CATCHUP_FAILED exit=%s queue_paused_at_index=%s\n' \
          "$(date -Is)" "$rc" "$start_index" | tee -a "$queue_dir/events.log"
        exit "$rc"
    fi
    printf '%s PREDICTION_FACTORY_CATCHUP_END before_index=%s\n' \
      "$(date -Is)" "$start_index" | tee -a "$queue_dir/events.log"
fi

failures=0
for index in "${!models[@]}"; do
    (( index < start_index )) && continue
    model=${models[$index]}
    log="$queue_dir/${model}.log"
    printf '%s START %s dataset=v1.1\n' "$(date -Is)" "$model" | tee -a "$queue_dir/events.log"
    feature_args=()
    if [[ "$model" == "catboost_stylometry" ]]; then
        feature_log="$queue_dir/catboost_stylometry_features.log"
        printf '%s FEATURE_START catboost_stylometry dataset=v1.1\n' \
          "$(date -Is)" | tee -a "$queue_dir/events.log"
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
          TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled \
          "$python" -m src.pipeline features build --dataset v1.1 --data-dir data/processed \
          --config configs/features/catboost_stylometry.yaml >"$feature_log" 2>&1
        rc=$?
        if (( rc != 0 )); then
            failures=$((failures + 1))
            printf '%s FEATURE_FAILED catboost_stylometry exit=%s log=%s skipped_model=true continuing_queue=true\n' \
              "$(date -Is)" "$rc" "$feature_log" | tee -a "$queue_dir/events.log"
            continue
        fi
        printf '%s FEATURE_END catboost_stylometry exit=0 log=%s\n' \
          "$(date -Is)" "$feature_log" | tee -a "$queue_dir/events.log"
        feature_args=(--features-config configs/features/catboost_stylometry.yaml)
    fi
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
      TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled \
      "$python" -m src.pipeline models train --model "$model" --dataset v1.1 \
      --config "$config_dir/${model}.yaml" "${feature_args[@]}" \
      --accept-config-defaults >"$log" 2>&1
    rc=$?
    if (( rc != 0 )); then
        failures=$((failures + 1))
        printf '%s FAILED %s exit=%s log=%s continuing_queue=true\n' \
          "$(date -Is)" "$model" "$rc" "$log" | tee -a "$queue_dir/events.log"
        continue
    fi
    printf '%s END %s exit=0 log=%s\n' "$(date -Is)" "$model" "$log" | tee -a "$queue_dir/events.log"
done

printf '%s QUEUE_FINISHED failure_count=%s\n' "$(date -Is)" "$failures" | tee -a "$queue_dir/events.log"
