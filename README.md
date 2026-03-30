# Curriculum Learning for SFT on MoE Models

This repository contains experiments for supervised fine-tuning with:

- data curriculum
- model curriculum for Mixture-of-Experts routing
- combined data + model curriculum
- regular LoRA or full fine-tuning

The code currently supports training and evaluation on:

- `openai/gsm8k`
- `allenai/ai2_arc`
- `allenai/sciq`

and includes config sets for these model families:

- `olmoe`
- `qwen`
- `phi`
- `granite`
- `gpt_oss`

## Repository Layout

```text
.
├── configs/                 # Train and eval YAMLs
├── core/                    # Shared utilities
├── curriculum/              # Curriculum state, datasets, callbacks
├── data/                    # Task adapters for GSM8K / ARC / SciQ
├── evaluation/              # Benchmark adapters and eval runner
├── models/                  # Model loading and MoE capability helpers
├── training/                # Training pipeline, collator, trainer
├── sft.py                   # Main training entry point
├── eval.py                  # Main evaluation entry point
├── run_all.sh               # Batch runner for OLMoE configs
└── environment.yml          # Conda environment
```

## Setup

Create the environment:

```bash
conda env create -f environment.yml
conda activate masters_thesis
```

If you use gated or large Hugging Face models, make sure you are authenticated:

```bash
huggingface-cli login
```

Optional:

- set `WANDB_API_KEY` if `report_to: wandb`
- set `HF_HOME` / cache paths if you want model downloads somewhere specific

## Training

Training is driven entirely by YAML configs:

```bash
python sft.py -c configs/qwen/train/arc_no_curriculum_lora.yaml
```

Important config switches:

- `use_data_curriculum`: enables curriculum sampling over example difficulty
- `use_model_curriculum`: enables stage-wise MoE top-k routing changes during training
- `use_lora`: when `true`, train a LoRA adapter; when `false`, do full fine-tuning

Checkpoint output:

- LoRA runs save to `.../final_adapter`
- full fine-tuning runs save to `.../final_model`

## Evaluation

Evaluation is also config-driven:

```bash
python eval.py -c configs/qwen/eval/arc_no_curriculum_lora.yaml
```

Supported checkpoint modes in eval configs:

- `base`
- `lora_adapter`
- `full_model`

Base model eval uses the model directly from `model_id`. Adapter eval loads the base model, attaches the LoRA adapter, and attempts `merge_and_unload()` before generation.

## Curriculum Modes

This repo supports two curriculum mechanisms:

### Data Curriculum

Training examples are sampled according to a curriculum state that progresses over training steps.

### Model Curriculum

For MoE models that expose `num_experts_per_tok`, the training callback adjusts routing top-k by curriculum stage. Early stages can use fewer experts, then ramp toward the model's default routing budget.

If `use_model_curriculum` is disabled but the model still supports MoE routing control, the code fixes routing to the model default.

## Supported Tasks

Training datasets:

- GSM8K via `openai/gsm8k`
- ARC via `allenai/ai2_arc`
- SciQ via `allenai/sciq`

Evaluation benchmarks:

- `gsm8k`
- `arc`
- `sciq`

## Config Layout

Each model family follows the same structure:

```text
configs/<model_family>/
├── train/
│   ├── <task>_no_curriculum_lora.yaml
│   ├── <task>_data_curriculum_lora.yaml
│   ├── <task>_model_curriculum_lora.yaml
│   └── <task>_full_curriculum_lora.yaml
└── eval/
    ├── <task>_base.yaml
    ├── <task>_no_curriculum_lora.yaml
    ├── <task>_data_curriculum_lora.yaml
    ├── <task>_model_curriculum_lora.yaml
    └── <task>_full_curriculum_lora.yaml
```

Current model families:

- `configs/olmoe`
- `configs/qwen`
- `configs/phi`
- `configs/granite`
- `configs/gpt_oss`

## Example Workflows

Train Qwen with LoRA on ARC without curriculum:

```bash
python sft.py -c configs/qwen/train/arc_no_curriculum_lora.yaml
```

Train Granite with full fine-tuning:

1. Open a Granite train config.
2. Set `use_lora: false`.
3. Run:

```bash
python sft.py -c configs/granite/train/gsm8k_no_curriculum_lora.yaml
```

Evaluate a saved adapter:

```bash
python eval.py -c configs/phi/eval/sciq_full_curriculum_lora.yaml
```

## Notes

- `run_all.sh` currently runs only the OLMoE LoRA experiments.
- `enable_task_metrics_during_training: false` is set in all train configs by default.
- Full fine-tuning, especially on larger or MoE models, will require substantially more GPU memory than LoRA.
- Some models may require additional Hugging Face access permissions depending on their license or availability.
