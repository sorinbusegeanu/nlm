#!/usr/bin/env bash
set -uo pipefail
cd /home/zodrak/nlm
mkdir -p artifacts/logs
LOG=artifacts/logs/added_models_training_queue.log
exec >>"$LOG" 2>&1
printf '\n[%s] Gemma4 text-only retry queued; waiting for the active model queue to release the GPU.\n' "$(date -Is)"
while systemctl --user is-active --quiet nlm-added-models-training-retry.service; do
  sleep 30
done
printf '[%s] Starting Gemma4 text-only LoRA training.\n' "$(date -Is)"
HF_HUB_OFFLINE=1 /home/zodrak/cuda13.x/bin/python -u -m src.pipeline models train \
  --model gemma4_e4b --dataset v0 --config configs/models/gemma4_e4b.yaml
