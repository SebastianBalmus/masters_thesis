#!/usr/bin/env bash

set -euo pipefail

MODELS=(olmoe qwen lfm2 gpt_oss)
TASKS=(gsm8k arc sciq)
VARIANTS=(base no_curriculum data_curriculum model_curriculum full_curriculum)

TOTAL_RUNS=$((${#MODELS[@]} * ${#TASKS[@]} * ${#VARIANTS[@]}))

echo "Total evaluations to run: ${TOTAL_RUNS}"

for model in "${MODELS[@]}"; do
  echo "Running sequential evaluations for model family: ${model}"

  for task in "${TASKS[@]}"; do
    echo "Running ${task} evaluation configs for ${model}"

    for variant in "${VARIANTS[@]}"; do
      if [[ "${variant}" == "base" ]]; then
        config_path="configs/${model}/eval/${task}_base.yaml"
      else
        config_path="configs/${model}/eval/${task}_${variant}_lora.yaml"
      fi

      echo "Starting evaluation: ${config_path}"
      python eval.py -c "${config_path}"
      echo "Finished evaluation: ${config_path}"
    done
  done

  echo "Completed all evaluation configs for ${model}"
done

echo "Completed all sequential evaluation runs for all model families."
