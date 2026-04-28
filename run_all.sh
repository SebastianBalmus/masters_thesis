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
printf "model\ttask\tmethod\tseed\tconfig_path\tstart_utc\tend_utc\ttotal_duration_seconds\ttraining_runtime_seconds\tvalidation_runtime_seconds\tinference_runtime_seconds\n" > "${TIMING_LOG}"

echo "Total trainings to run: ${TOTAL_RUNS}"
echo "Timing log: ${TIMING_LOG}"

for model in "${MODELS[@]}"; do
  echo "Running sequential training for model family: ${model}"

  for task in "${TASKS[@]}"; do
    echo "Running ${task} training configs for ${model}"

    for method in "${METHODS[@]}"; do
      config_path="configs/${model}/train/${task}_${method}_lora.yaml"
      output_dir=$(awk -F': ' '/^output_dir:/ {print $2; exit}' "${config_path}")
      for seed in "${SEEDS[@]}"; do
        echo "Starting training: ${config_path} (seed=${seed})"
        start_epoch=$(date +%s)
        start_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        python sft.py -c "${config_path}" --seed "${seed}"
        end_epoch=$(date +%s)
        end_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        total_duration_seconds=$((end_epoch - start_epoch))

        summary_path="${output_dir}__seed_${seed}/run_summary.json"
        if [[ ! -f "${summary_path}" ]]; then
          echo "Missing run summary: ${summary_path}" >&2
          exit 1
        fi

        training_runtime_seconds=$(jq -r '.training_runtime_seconds // ""' "${summary_path}")
        validation_runtime_seconds=$(jq -r '.validation_runtime_seconds // ""' "${summary_path}")
        inference_runtime_seconds=$(jq -r '.inference_runtime_seconds // ""' "${summary_path}")

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
          "${model}" \
          "${task}" \
          "${method}" \
          "${seed}" \
          "${config_path}" \
          "${start_utc}" \
          "${end_utc}" \
          "${total_duration_seconds}" \
          "${training_runtime_seconds}" \
          "${validation_runtime_seconds}" \
          "${inference_runtime_seconds}" >> "${TIMING_LOG}"
        echo "Finished training: ${config_path} (seed=${seed}, total=${total_duration_seconds}s, train=${training_runtime_seconds}s, val=${validation_runtime_seconds}s, test=${inference_runtime_seconds}s)"
      done
    done
  done

  echo "Completed all training configs for ${model}"
done

echo "Completed all sequential training runs for all model families."
