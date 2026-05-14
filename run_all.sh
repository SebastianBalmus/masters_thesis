#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: $0 {lora|full_ft} [--delete-checkpoints]" >&2
  exit 1
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
fi

FT_MODE="$1"
DELETE_CHECKPOINTS=false

if [[ $# -eq 2 ]]; then
  case "$2" in
    --delete-checkpoints)
      DELETE_CHECKPOINTS=true
      ;;
    *)
      usage
      ;;
  esac
fi

case "${FT_MODE}" in
  lora)
    CONFIG_SUBDIR="lora"
    CONFIG_SUFFIX="lora"
    ;;
  full_ft)
    CONFIG_SUBDIR="full_ft"
    CONFIG_SUFFIX="full"
    ;;
  *)
    usage
    ;;
esac

MODELS=(olmoe qwen lfm2 gpt_oss)
FSDP_MODELS=(qwen gpt_oss)
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
TIMING_LOG="${LOG_DIR}/training_times_${FT_MODE}.tsv"

mkdir -p "${LOG_DIR}"
printf "ft_mode\tmodel\ttask\tmethod\tseed\tconfig_path\tstart_utc\tend_utc\ttotal_duration_seconds\ttraining_runtime_seconds\tvalidation_runtime_seconds\tinference_runtime_seconds\n" > "${TIMING_LOG}"

echo "Total trainings to run: ${TOTAL_RUNS}"
echo "Fine-tuning mode: ${FT_MODE}"
echo "Timing log: ${TIMING_LOG}"
echo "Delete checkpoints after each run: ${DELETE_CHECKPOINTS}"
echo "FSDP processes for large full_ft runs: ${FSDP_NUM_PROCESSES:-2}"

uses_fsdp_for_model() {
  local model="$1"
  if [[ "${FT_MODE}" != "full_ft" ]]; then
    return 1
  fi

  for fsdp_model in "${FSDP_MODELS[@]}"; do
    if [[ "${model}" == "${fsdp_model}" ]]; then
      return 0
    fi
  done

  return 1
}

for model in "${MODELS[@]}"; do
  echo "Running sequential training for model family: ${model}"

  for task in "${TASKS[@]}"; do
    echo "Running ${task} training configs for ${model}"

    for method in "${METHODS[@]}"; do
      config_path="configs/${model}/train/${CONFIG_SUBDIR}/${task}_${method}_${CONFIG_SUFFIX}.yaml"
      if [[ ! -f "${config_path}" ]]; then
        echo "Missing config: ${config_path}" >&2
        exit 1
      fi

      output_dir=$(awk -F': ' '/^output_dir:/ {print $2; exit}' "${config_path}")
      for seed in "${SEEDS[@]}"; do
        echo "Starting training: ${config_path} (seed=${seed})"
        start_epoch=$(date +%s)
        start_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        if uses_fsdp_for_model "${model}"; then
          accelerate launch --num_processes "${FSDP_NUM_PROCESSES:-2}" sft.py -c "${config_path}" --seed "${seed}" --fsdp
        else
          python sft.py -c "${config_path}" --seed "${seed}"
        fi
        end_epoch=$(date +%s)
        end_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        total_duration_seconds=$((end_epoch - start_epoch))

        run_output_dir="${output_dir}__seed_${seed}"
        summary_path="${run_output_dir}/run_summary.json"
        if [[ ! -f "${summary_path}" ]]; then
          echo "Missing run summary: ${summary_path}" >&2
          exit 1
        fi

        training_runtime_seconds=$(jq -r '.training_runtime_seconds // ""' "${summary_path}")
        validation_runtime_seconds=$(jq -r '.validation_runtime_seconds // ""' "${summary_path}")
        inference_runtime_seconds=$(jq -r '.inference_runtime_seconds // ""' "${summary_path}")

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
          "${FT_MODE}" \
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

        if [[ "${DELETE_CHECKPOINTS}" == true ]]; then
          if [[ -n "${run_output_dir}" && "${run_output_dir}" == ckpt/* && -d "${run_output_dir}" ]]; then
            echo "Deleting checkpoint directory: ${run_output_dir}"
            rm -rf -- "${run_output_dir}"
          else
            echo "Refusing to delete unexpected checkpoint path: ${run_output_dir}" >&2
            exit 1
          fi
        fi
      done
    done
  done

  echo "Completed all training configs for ${model}"
done

echo "Completed all sequential ${FT_MODE} training runs for all model families."
