#!/usr/bin/env bash

set -euo pipefail

wait_for_gpu_free() {
  local threshold_mb=2000
  while true; do
    used_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1)
    if [[ "${used_mb}" -lt "${threshold_mb}" ]]; then
      break
    fi
    echo "GPU memory still high (${used_mb} MB). Waiting..."
    sleep 5
  done
}

MODELS=(olmoe qwen lfm2 gpt_oss)
TASKS=(arc sciq gsm8k)
VARIANTS=(
  fixed_k_max
  fixed_k_1
  linear_k_1_to_topk
  warmup
  linear_mid_start
  frontloaded
  backloaded
  jump_warmup
)

TOTAL_RUNS=$((${#MODELS[@]} * ${#TASKS[@]} * ${#VARIANTS[@]}))

echo "Total evaluations to run: ${TOTAL_RUNS}"

for model in "${MODELS[@]}"; do
  echo "Running sequential evaluations for model family: ${model}"

  for task in "${TASKS[@]}"; do
    echo "Running ${task} evaluation configs for ${model}"

    for variant in "${VARIANTS[@]}"; do
      config_path="configs/${model}/eval/${task}_${variant}_lora.yaml"
      wait_for_gpu_free
      echo "Starting evaluation: ${config_path}"
      python eval.py -c "${config_path}"
      sleep 10
      echo "Finished evaluation: ${config_path}"
      wait_for_gpu_free
    done
  done

  echo "Completed all evaluation configs for ${model}"
done

echo "Completed all sequential evaluation runs for all model families."
