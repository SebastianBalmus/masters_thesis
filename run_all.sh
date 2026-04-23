#!/usr/bin/env bash

set -euo pipefail

MODELS=(olmoe qwen lfm2 gpt_oss)
TASKS=(arc sciq gsm8k)
SEEDS=(42 123 999)
METHODS=(
  fixed_k_max
  fixed_k_1
  linear_k_1_to_topk
  frontloaded
  backloaded
  linear_mid_start
  warmup
  jump_warmup
)

TOTAL_RUNS=$((${#MODELS[@]} * ${#TASKS[@]} * ${#METHODS[@]} * ${#SEEDS[@]}))
LOG_DIR="logs"
TIMING_LOG="${LOG_DIR}/training_times.tsv"

mkdir -p "${LOG_DIR}"
printf "model\ttask\tmethod\tseed\tconfig_path\tstart_utc\tend_utc\tduration_seconds\n" > "${TIMING_LOG}"

echo "Total trainings to run: ${TOTAL_RUNS}"
echo "Timing log: ${TIMING_LOG}"

for model in "${MODELS[@]}"; do
  echo "Running sequential training for model family: ${model}"

  for task in "${TASKS[@]}"; do
    echo "Running ${task} training configs for ${model}"

    for method in "${METHODS[@]}"; do
      config_path="configs/${model}/train/${task}_${method}_lora.yaml"
      for seed in "${SEEDS[@]}"; do
        echo "Starting training: ${config_path} (seed=${seed})"
        start_epoch=$(date +%s)
        start_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        python sft.py -c "${config_path}" --seed "${seed}"
        end_epoch=$(date +%s)
        end_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        duration_seconds=$((end_epoch - start_epoch))
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
          "${model}" \
          "${task}" \
          "${method}" \
          "${seed}" \
          "${config_path}" \
          "${start_utc}" \
          "${end_utc}" \
          "${duration_seconds}" >> "${TIMING_LOG}"
        echo "Finished training: ${config_path} (seed=${seed}, ${duration_seconds}s)"
      done
    done
  done

  echo "Completed all training configs for ${model}"
done

echo "Completed all sequential training runs for all model families."
