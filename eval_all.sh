#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: $0 {lora|full_ft}" >&2
  exit 1
}

if [[ $# -ne 1 ]]; then
  usage
fi

FT_MODE="$1"
case "${FT_MODE}" in
  lora)
    CONFIG_SUBDIR="lora"
    CONFIG_SUFFIX="lora"
    CHECKPOINT_MODE="lora_adapter"
    FINAL_DIR="final_adapter"
    ;;
  full_ft)
    CONFIG_SUBDIR="full_ft"
    CONFIG_SUFFIX="full"
    CHECKPOINT_MODE="full_model"
    FINAL_DIR="final_model"
    ;;
  *)
    usage
    ;;
esac

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
SEEDS=(42 123 999)
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

TOTAL_RUNS=$((${#MODELS[@]} * ${#TASKS[@]} * ${#VARIANTS[@]} * ${#SEEDS[@]}))

echo "Total evaluations to run: ${TOTAL_RUNS}"
echo "Fine-tuning mode: ${FT_MODE}"

for model in "${MODELS[@]}"; do
  echo "Running sequential evaluations for model family: ${model}"

  for task in "${TASKS[@]}"; do
    echo "Running ${task} evaluation configs for ${model}"

    for variant in "${VARIANTS[@]}"; do
      config_path="configs/${model}/eval/${CONFIG_SUBDIR}/${task}_${variant}_${CONFIG_SUFFIX}.yaml"
      if [[ ! -f "${config_path}" ]]; then
        echo "Missing config: ${config_path}" >&2
        exit 1
      fi

      run_name_base=$(awk -F': ' '/^run_name:/ {print $2; exit}' "${config_path}")
      if [[ -z "${run_name_base}" ]]; then
        echo "Missing run_name in config: ${config_path}" >&2
        exit 1
      fi

      for seed in "${SEEDS[@]}"; do
        run_name="${run_name_base}__seed_${seed}"
        checkpoint_path="ckpt/${run_name}/${FINAL_DIR}"

        wait_for_gpu_free
        echo "Starting evaluation: ${config_path} (seed=${seed}, checkpoint=${checkpoint_path})"
        python eval.py \
          -c "${config_path}" \
          --seed "${seed}" \
          --run-name "${run_name}" \
          --checkpoint-mode "${CHECKPOINT_MODE}" \
          --checkpoint-path "${checkpoint_path}"
        sleep 10
        echo "Finished evaluation: ${config_path} (seed=${seed})"
        wait_for_gpu_free
      done
    done
  done

  echo "Completed all evaluation configs for ${model}"
done

echo "Completed all sequential ${FT_MODE} evaluation runs for all model families."
