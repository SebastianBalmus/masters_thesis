#!/usr/bin/env bash

set -euo pipefail

MODELS=(olmoe qwen gpt_oss)
TASKS=(gsm8k arc sciq)
METHODS=(no_curriculum model_curriculum data_curriculum full_curriculum)

TOTAL_RUNS=$((${#MODELS[@]} * ${#TASKS[@]} * ${#METHODS[@]}))
LOG_DIR="logs"
TIMING_LOG="${LOG_DIR}/training_times.tsv"

mkdir -p "${LOG_DIR}"
printf "model\ttask\tmethod\tconfig_path\tstart_utc\tend_utc\tduration_seconds\n" > "${TIMING_LOG}"

echo "Total trainings to run: ${TOTAL_RUNS}"
echo "Timing log: ${TIMING_LOG}"

for model in "${MODELS[@]}"; do
  echo "Running sequential training for model family: ${model}"

  for task in "${TASKS[@]}"; do
    echo "Running ${task} training configs for ${model}"

    for method in "${METHODS[@]}"; do
      config_path="configs/${model}/train/${task}_${method}_lora.yaml"
      echo "Starting training: ${config_path}"
      start_epoch=$(date +%s)
      start_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
      python sft.py -c "${config_path}"
      end_epoch=$(date +%s)
      end_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
      duration_seconds=$((end_epoch - start_epoch))
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${model}" \
        "${task}" \
        "${method}" \
        "${config_path}" \
        "${start_utc}" \
        "${end_utc}" \
        "${duration_seconds}" >> "${TIMING_LOG}"
      echo "Finished training: ${config_path} (${duration_seconds}s)"
    done
  done

  echo "Completed all training configs for ${model}"
done

echo "Completed all sequential training runs for all model families."
