# Curriculum Learning for SFT on MoE Models

This repository contains a config-driven training and evaluation pipeline for supervised fine-tuning causal language models on reasoning and QA benchmarks, with a focus on curriculum learning for Mixture-of-Experts models.

The codebase supports:

- fixed MoE routing at the model default top-k
- fixed MoE routing at `k=1`
- linear MoE routing curriculum from `k=1` to the model default top-k
- linear MoE routing curriculum from `k=floor(topk/2)` to the model default top-k
- warmup routing from `k=1` to the model default top-k during an initial configurable fraction of training
- frontloaded MoE routing curriculum with less time at low `k` and more time at high `k`
- backloaded MoE routing curriculum with more time at low `k` and less time at high `k`
- jump warmup routing that stays at `k=1` for an initial configurable fraction of training, then switches to the model default top-k
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
├── curriculum/              # Model-curriculum state and callbacks
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
python sft.py -c configs/qwen/train/arc_fixed_k_max_lora.yaml
```

Each training config specifies:

- the base model via `model_id`
- the dataset via `dataset_id`
- the routing method via `routing_method`
- whether LoRA is enabled via `use_lora`
- optimization and evaluation settings
- checkpoint and logging destinations

Representative training fields:

```yaml
model_id: Qwen/Qwen1.5-MoE-A2.7B-Chat
dataset_id: allenai/ai2_arc

routing_method: fixed_k_max

use_lora: true
lora_config:
  r: 32
  lora_alpha: 64
  target_modules: all-linear
  lora_dropout: 0.05

per_device_train_batch_size: 4
per_device_eval_batch_size: 4
gradient_accumulation_steps: 1

output_dir: ckpt/qwen_arc_fixed_k_max_lora
```

### Training Outputs

By default, training writes into the configured `output_dir`, typically under `ckpt/`.

Final artifacts are saved as:

- `.../final_adapter` for LoRA runs
- `.../final_model` for full fine-tuning runs

Intermediate checkpoints are also saved according to the config's `save_steps`.

### Curriculum Behavior

This repository implements eight MoE routing methods:

- `fixed_k_max`: keep routing fixed at the model's default `num_experts_per_tok`
- `fixed_k_1`: keep routing fixed at `k=1`
- `linear_k_1_to_topk`: start at `k=1` and increase one integer at a time until the model's default top-k, with equal training-step allocation per `k`
- `linear_mid_start`: same as `linear_k_1_to_topk`, but starts at `k=floor(topk/2)`
- `warmup`: linearly increase from `k=1` to the model default top-k during the initial `routing_transition_ratio` fraction of steps, then stay at the default top-k
- `frontloaded`: use stage weights `[1, 2, ..., K]`, so training spends less time at low `k` and more time near the default top-k
- `backloaded`: use stage weights `[K, K-1, ..., 1]`, so training spends more time at low `k` and less time near the default top-k
- `jump_warmup`: keep `k=1` during the initial `routing_transition_ratio` fraction of steps, then jump to the model default top-k

Training runs for exactly one epoch on the train split. During that epoch the trainer:

- evaluates periodically on validation
- saves checkpoints periodically
- tracks the best validation accuracy and step

After training it evaluates both the best-validation checkpoint and the final checkpoint on the test split, then writes a run summary with:

- `best_val_accuracy`
- `best_step`
- `test_accuracy_at_best`
- `test_accuracy_at_final`
- `train_runtime`
- `final_step`
- the routing `k` value recorded at each validation evaluation step

### Task Metrics During Training

This protocol requires task-specific validation accuracy, so training always runs the lightweight generative evaluation callback on the validation split.

## Evaluation

Evaluation is also config-driven:

```bash
python eval.py -c configs/qwen/eval/arc_fixed_k_max_lora.yaml
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
  path: ckpt/qwen_arc_fixed_k_max_lora/final_adapter

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
│   ├── <task>_fixed_k_max_lora.yaml
│   ├── <task>_fixed_k_1_lora.yaml
│   ├── <task>_linear_k_1_to_topk_lora.yaml
│   ├── <task>_linear_mid_start_lora.yaml
│   ├── <task>_warmup_lora.yaml
│   ├── <task>_frontloaded_lora.yaml
│   ├── <task>_backloaded_lora.yaml
│   └── <task>_jump_warmup_lora.yaml
└── eval/
    ├── <task>_base.yaml
    ├── <task>_fixed_k_max_lora.yaml
    ├── <task>_fixed_k_1_lora.yaml
    ├── <task>_linear_k_1_to_topk_lora.yaml
    ├── <task>_linear_mid_start_lora.yaml
    ├── <task>_warmup_lora.yaml
    ├── <task>_frontloaded_lora.yaml
    ├── <task>_backloaded_lora.yaml
    └── <task>_jump_warmup_lora.yaml
```

The naming convention is consistent across the repo:

- `fixed_k_max`: fixed routing at the model default top-k
- `fixed_k_1`: fixed routing at `k=1`
- `linear_k_1_to_topk`: linear top-k curriculum from `1` to the model default
- `linear_mid_start`: linear top-k curriculum from `floor(topk/2)` to the model default
- `warmup`: linear warmup from `1` to the model default during the initial `routing_transition_ratio`
- `frontloaded`: weighted top-k curriculum with stage weights `[1, 2, ..., K]`
- `backloaded`: weighted top-k curriculum with stage weights `[K, K-1, ..., 1]`
- `jump_warmup`: hold `k=1` during the initial `routing_transition_ratio`, then jump to the model default
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
- methods: `fixed_k_max`, `fixed_k_1`, `linear_k_1_to_topk`, `linear_mid_start`, `warmup`, `frontloaded`, `backloaded`, `jump_warmup`

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
- variants: `base`, `fixed_k_max`, `fixed_k_1`, `linear_k_1_to_topk`, `linear_mid_start`, `warmup`, `frontloaded`, `backloaded`, `jump_warmup`

## Typical Workflows

Train one configuration:

```bash
python sft.py -c configs/gpt_oss/train/gsm8k_linear_k_1_to_topk_lora.yaml
```

Evaluate the resulting LoRA adapter:

```bash
python eval.py -c configs/gpt_oss/eval/gsm8k_linear_k_1_to_topk_lora.yaml
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
