# Curriculum Learning for SFT on MoE Models

This repository contains a config-driven training and evaluation pipeline for supervised fine-tuning mixture-of-experts language models on three reasoning benchmarks. The tracked code focuses on comparing fixed and curriculum-based top-k routing schedules under LoRA fine-tuning, plus base-model and post-training evaluation.

## Current Scope

The checked-in experiment matrix covers:

- model families: `olmoe`, `qwen`, `lfm2`, `gpt_oss`
- tasks: `arc`, `gsm8k`, `sciq`
- training routing methods: `fixed_k_max`, `fixed_k_1`, `linear_k_1_to_topk`, `linear_mid_start`, `warmup`, `frontloaded`, `backloaded`, `jump_warmup`
- evaluation variants: the same eight routing methods plus `base`

The repository currently includes:

- training and evaluation entrypoints in `sft.py` and `eval.py`
- reusable modules for curriculum control, model loading, dataset adapters, evaluation runners, and training callbacks
- YAML configs under `configs/` for every tracked model family, task, and routing variant
- batch scripts for running the full train and eval matrices
- one analysis notebook: `notebooks/eval_metric_runtime_deltas.ipynb`
- a Conda environment definition in `environment.yml`

Generated outputs such as checkpoints, logs, W&B state, and evaluation result directories are intentionally not part of the tracked repo scope.

## Repository Layout

```text
.
├── configs/                 # Train and eval YAMLs grouped by model family
├── core/                    # Shared utilities such as seeding
├── curriculum/              # Routing state, schedules, and trainer callbacks
├── data/                    # Dataset adapters and formatting for train/eval splits
├── evaluation/              # Benchmark adapters, model loading, and eval runner
├── models/                  # Model factory and MoE capability helpers
├── training/                # SFT pipeline, trainer, collator, and metrics callbacks
├── notebooks/               # Analysis notebook for eval/runtime deltas
├── sft.py                   # Main training entry point
├── eval.py                  # Main evaluation entry point
├── run_all.sh               # Sequential runner for all tracked training configs
├── eval_all.sh              # Sequential runner for all tracked eval configs
└── environment.yml          # Conda environment definition
```

## Training

Training is driven by YAML configs:

```bash
python sft.py -c configs/qwen/train/arc_fixed_k_max_lora.yaml
```

The training stack supports:

- LoRA-based fine-tuning through the checked-in configs
- dynamic MoE top-k routing schedules during training
- periodic validation with generative task metrics
- automatic post-training test evaluation for best and final checkpoints

The tracked train configs cover all combinations of:

- 4 model families
- 3 tasks
- 8 routing methods

## Evaluation

Evaluation is also config-driven:

```bash
python eval.py -c configs/qwen/eval/arc_fixed_k_max_lora.yaml
```

The evaluation entrypoint supports:

- base-model evaluation
- LoRA adapter evaluation
- full-model checkpoint evaluation
- benchmark-specific prompting and answer extraction for `arc`, `gsm8k`, and `sciq`

`eval.py` currently expects CUDA and raises an error when no GPU is available.

## Batch Workflows

Run the full tracked training matrix:

```bash
./run_all.sh
```

Run the full tracked evaluation matrix:

```bash
./eval_all.sh
```

These scripts iterate across the checked-in model families, tasks, and routing variants defined in `configs/`.

## Setup

Create the environment:

```bash
conda env create -f environment.yml
conda activate disertatie
```

If you need gated Hugging Face models or datasets:

```bash
huggingface-cli login
```

If you log runs to Weights & Biases:

```bash
export WANDB_API_KEY=...
```

## Notes

- The checked-in configs are LoRA-oriented, but the training code also supports non-LoRA runs via config changes.
- Curriculum routing only applies to models that expose MoE top-k routing in their config.
- The tracked notebook is for local analysis of evaluation metrics and runtime deltas after experiments have been run.
