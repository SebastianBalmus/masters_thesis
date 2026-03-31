# Curriculum Learning for SFT on MoE Models

This repository contains a config-driven training and evaluation pipeline for supervised fine-tuning causal language models on reasoning and QA benchmarks, with a focus on curriculum learning for Mixture-of-Experts models.

The codebase supports:

- standard SFT without curriculum
- data curriculum over example difficulty
- model curriculum via stage-wise MoE top-k routing
- combined data + model curriculum
- LoRA-based fine-tuning
- base-model and fine-tuned-model evaluation

## What Is In The Repo

The current experiment matrix is built around four model families:

- `olmoe`
- `qwen`
- `lfm2`
- `gpt_oss`

and three datasets / benchmarks:

- `openai/gsm8k`
- `allenai/ai2_arc`
- `allenai/sciq`

The repository already contains:

- train configs under `configs/<model_family>/train/`
- eval configs under `configs/<model_family>/eval/`
- training checkpoints under `ckpt/`
- evaluation outputs under `eval_results/`
- batch scripts for running the full train or eval matrix

## Repository Layout

```text
.
├── configs/                 # Train and eval YAMLs grouped by model family
├── core/                    # Shared utilities such as seeding
├── curriculum/              # Curriculum state, iterable datasets, callbacks
├── data/                    # Dataset adapters for GSM8K, ARC, SciQ
├── evaluation/              # Benchmark adapters, model loading, eval runner
├── models/                  # Model loading and MoE capability helpers
├── training/                # Training pipeline, collator, trainer, callbacks
├── sft.py                   # Main training entry point
├── eval.py                  # Main evaluation entry point
├── run_all.sh               # Sequential training runner for all configured models
├── eval_all.sh              # Sequential evaluation runner for all configured models
├── environment.yml          # Conda environment
├── ckpt/                    # Saved checkpoints and final artifacts
├── eval_results/            # Saved predictions and metrics
├── logs/                    # Auxiliary script outputs
└── wandb/                   # Local Weights & Biases run metadata
```

## Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate masters_thesis
```

If you plan to use Hugging Face-hosted models or datasets, authenticate first:

```bash
huggingface-cli login
```

If training logs are sent to Weights & Biases, also export:

```bash
export WANDB_API_KEY=...
```

Useful optional environment variables:

- `HF_HOME` for a custom Hugging Face cache location
- `TRANSFORMERS_CACHE` if you want model weights on a specific disk

## Training

Training is fully driven by YAML configs:

```bash
python sft.py -c configs/qwen/train/arc_no_curriculum_lora.yaml
```

Each training config specifies:

- the base model via `model_id`
- the dataset via `dataset_id`
- whether data curriculum is enabled via `use_data_curriculum`
- whether model curriculum is enabled via `use_model_curriculum`
- whether LoRA is enabled via `use_lora`
- optimization and evaluation settings
- checkpoint and logging destinations

Representative training fields:

```yaml
model_id: Qwen/Qwen1.5-MoE-A2.7B-Chat
dataset_id: allenai/ai2_arc

use_data_curriculum: false
use_model_curriculum: false

use_lora: true
lora_config:
  r: 32
  lora_alpha: 64
  target_modules: all-linear
  lora_dropout: 0.05

per_device_train_batch_size: 4
per_device_eval_batch_size: 4
gradient_accumulation_steps: 1

output_dir: ckpt/qwen_arc_no_curriculum_lora
```

### Training Outputs

By default, training writes into the configured `output_dir`, typically under `ckpt/`.

Final artifacts are saved as:

- `.../final_adapter` for LoRA runs
- `.../final_model` for full fine-tuning runs

Intermediate checkpoints are also saved according to the config's `save_steps`.

### Curriculum Behavior

This repository implements two distinct curriculum mechanisms.

`use_data_curriculum: true`

- the training dataset is wrapped in a curriculum-aware iterable dataset
- sampling progresses over difficulty levels during training

`use_model_curriculum: true`

- only valid for models whose config exposes `num_experts_per_tok`
- routing top-k is adjusted stage by stage during training
- if enabled on a model that does not support MoE routing control, training raises an error

If model curriculum is disabled but the model still supports MoE routing control, the training code fixes routing to the model's default top-k.

### Task Metrics During Training

Train configs expose `enable_task_metrics_during_training`.

- when `true`, the trainer runs a lightweight generative evaluation callback on the validation split during training
- when `false`, training skips this extra benchmark-style validation loop

The current configs set this to `false` by default.

## Evaluation

Evaluation is also config-driven:

```bash
python eval.py -c configs/qwen/eval/arc_no_curriculum_lora.yaml
```

The evaluation entry point currently expects CUDA and will raise an error if no GPU is available.

Each eval config contains:

- `model_id` for the base model
- `checkpoint.mode` describing what to load
- `checkpoint.path` when evaluating a saved adapter or full model
- an `eval` block with benchmark and generation settings

Representative evaluation fields:

```yaml
model_id: Qwen/Qwen1.5-MoE-A2.7B-Chat

checkpoint:
  mode: lora_adapter
  path: ckpt/qwen_arc_no_curriculum_lora/final_adapter

eval:
  benchmark: arc
  split: test
  results_dir: eval_results
  batch_size: 16
  max_new_tokens: 8
  sort_by_length: true
  compile_model: false
  attn_implementation: sdpa
```

### Supported Checkpoint Modes

- `base`: evaluate the untouched base model from `model_id`
- `lora_adapter`: load the base model, attach the adapter, and attempt `merge_and_unload()`
- `full_model`: load a fully fine-tuned checkpoint directly from `checkpoint.path`

### Evaluation Outputs

Evaluation results are written to:

```text
eval_results/<benchmark>/<run_name>/
```

Each run produces:

- `predictions.jsonl`: prompt, generation, extracted answer, gold answer, correctness
- `metrics.json`: aggregate metrics and metadata such as model id, checkpoint mode, and output paths

If `run_name` is omitted, it is derived automatically from the checkpoint mode and path.

### Eval Batch Sizes

The current eval configs are normalized by dataset across all model families:

- `arc`: `batch_size: 16`
- `sciq`: `batch_size: 16`
- `gsm8k`: `batch_size: 8`

## Config Layout

Each model family follows the same directory structure:

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

The naming convention is consistent across the repo:

- `no_curriculum`: plain LoRA fine-tuning
- `data_curriculum`: curriculum over example difficulty only
- `model_curriculum`: curriculum over MoE routing only
- `full_curriculum`: data curriculum plus model curriculum
- `base`: base-model evaluation only

## Batch Scripts

Two convenience scripts are available at repo root.

Train the full configured matrix:

```bash
./run_all.sh
```

This runs sequential training over:

- models: `olmoe`, `qwen`, `lfm2`, `gpt_oss`
- tasks: `gsm8k`, `arc`, `sciq`
- methods: `no_curriculum`, `model_curriculum`, `data_curriculum`, `full_curriculum`

and writes timing data to:

```text
logs/training_times.tsv
```

Evaluate the full configured matrix:

```bash
./eval_all.sh
```

This runs sequential evaluation over:

- models: `olmoe`, `qwen`, `lfm2`, `gpt_oss`
- tasks: `gsm8k`, `arc`, `sciq`
- variants: `base`, `no_curriculum`, `data_curriculum`, `model_curriculum`, `full_curriculum`

## Typical Workflows

Train one configuration:

```bash
python sft.py -c configs/gpt_oss/train/gsm8k_full_curriculum_lora.yaml
```

Evaluate the resulting LoRA adapter:

```bash
python eval.py -c configs/gpt_oss/eval/gsm8k_full_curriculum_lora.yaml
```

Evaluate the base model for comparison:

```bash
python eval.py -c configs/gpt_oss/eval/gsm8k_base.yaml
```

Run every training config:

```bash
./run_all.sh
```

Run every eval config:

```bash
./eval_all.sh
```

## Notes And Constraints

- Evaluation currently requires CUDA.
- LoRA is the default mode used by the checked-in train configs.
- Full fine-tuning is still supported by setting `use_lora: false`, but it will require substantially more memory.
- Model curriculum only applies to models whose config exposes `num_experts_per_tok`.
- Some model families may require Hugging Face access approval depending on license and availability.
- The repository already contains generated checkpoints, eval outputs, logs, and local `wandb/` state; if you want a cleaner working tree for reproduction, account for those directories.
