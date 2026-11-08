#!/usr/bin/env bash
set -u

cd /home/zodrak/nlm
python=/home/zodrak/cuda13.x/bin/python
queue_dir=/home/zodrak/nlm/artifacts/runs/v1_1_llm_queue_20261103
config_dir="$queue_dir/configs"
mkdir -p "$config_dir"

"$python" - "$config_dir" <<'PY'
import sys
from pathlib import Path
import yaml

out = Path(sys.argv[1])
models = [
    "embeddinggemma2", "gemma4_e4b", "granite4_350m", "llama32_3b",
    "ministral3_3b", "qwen06", "qwen17", "qwen4", "smollm3_3b", "ultrabert380",
]
for model in models:
    source = Path("configs/models") / f"{model}.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["warmup_epochs"] = int(config.get("epochs", 10))
    config.update({
        "epochs": 100,
        "early_stopping_patience": 3,
        "max_batch_size": 1024,
        "learning_rate_decay": 0.8,
        "lr_scheduler_type": "epoch_decay",
        "checkpoint_selection": "highest_validation_accuracy",
        "local_files_only": True,
    })
    (out / f"{model}.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

printf '%s QUEUE_START dataset=v1.1 max_epochs=100 early_stopping_patience=3 decay=0.8 offline=true\n' "$(date -Is)" | tee -a "$queue_dir/events.log"
printf '%s\n' 'embeddinggemma2 gemma4_e4b granite4_350m llama32_3b ministral3_3b qwen06 qwen17 qwen4 smollm3_3b ultrabert380' > "$queue_dir/model_order.txt"
failures=0

for model in embeddinggemma2 gemma4_e4b granite4_350m llama32_3b ministral3_3b qwen06 qwen17 qwen4 smollm3_3b ultrabert380; do
    log="$queue_dir/${model}.log"
    printf '%s START %s dataset=v1.1\n' "$(date -Is)" "$model" | tee -a "$queue_dir/events.log"
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
      TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled \
      "$python" -m src.pipeline models train --model "$model" --dataset v1.1 \
      --config "$config_dir/${model}.yaml" --accept-config-defaults >"$log" 2>&1
    rc=$?
    if (( rc == 0 )); then
        printf '%s COMPLETE %s exit=0 log=%s\n' "$(date -Is)" "$model" "$log" | tee -a "$queue_dir/events.log"
    else
        failures=$((failures + 1))
        printf '%s FAILED %s exit=%s log=%s continuing_queue=true\n' \
          "$(date -Is)" "$model" "$rc" "$log" | tee -a "$queue_dir/events.log"
    fi
done

printf '%s QUEUE_FINISHED failure_count=%s\n' "$(date -Is)" "$failures" | tee -a "$queue_dir/events.log"
