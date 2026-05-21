# Routing Curriculum Experiments for MoE SFT

This repository runs supervised fine-tuning and evaluation experiments for
mixture-of-experts language models. The goal is to compare different expert
routing schedules during fine-tuning, across multiple models and reasoning
benchmarks, using either LoRA adapters or full fine-tuning.

The experiment matrix currently covers:

- model families: `olmoe`, `qwen`, `lfm2`, `gpt_oss`
- benchmarks: `arc`, `gsm8k`, `sciq`
- routing methods: `fixed_k_max`, `fixed_k_1`, `linear_k_1_to_topk`,
  `frontloaded`, `backloaded`, `linear_mid_start`, `warmup`, `jump_warmup`
- fine-tuning modes: `lora` and `full_ft`
- seeds: `42`, `123`, `999`

## Repository Layout

```text
.
├── configs/                 # Train/eval YAMLs grouped by model and FT mode
│   ├── <model>/train/lora/
│   ├── <model>/train/full_ft/
│   ├── <model>/eval/lora/
│   └── <model>/eval/full_ft/
├── core/                    # Shared utilities: seeding, router-logit context
├── curriculum/              # Dynamic MoE top-k routing schedules and callbacks
├── data/                    # Dataset adapters for training splits
├── evaluation/              # Eval model loading, benchmark adapters, runner
├── models/                  # Model construction and MoE helpers
├── training/                # SFT pipeline, trainer, collator, metrics callbacks
├── notebooks/               # Analysis notebook for metrics and runtime deltas
├── sft.py                   # Single-config training entry point
├── eval.py                  # Single-config evaluation entry point
├── run_all.sh               # Full training-matrix runner
├── eval_all.sh              # Full evaluation-matrix runner
└── environment.yml          # Conda environment definition
```

## Setup

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate disertatie
```

Log in to Hugging Face if the selected model or dataset requires access:

```bash
huggingface-cli login
```

Weights & Biases logging is controlled by the training configs. If enabled,
set the usual W&B environment variables:

```bash
export WANDB_API_KEY=...
```

`eval.py` expects CUDA. The batch scripts also assume `nvidia-smi` is available.

## Configs

Configs are split by model, phase, and fine-tuning mode:

```text
configs/<model>/train/lora/*_lora.yaml
configs/<model>/train/full_ft/*_full.yaml
configs/<model>/eval/lora/*_lora.yaml
configs/<model>/eval/full_ft/*_full.yaml
```

Training configs define the dataset, routing method, fine-tuning mode,
hyperparameters, W&B run name, and output directory prefix.

Evaluation configs define the benchmark, model id, run-name prefix, checkpoint
mode, checkpoint path template, and generation settings. The batch eval script
overrides run names and checkpoint paths per seed so results are stored as
seeded runs.

## Training

Run one config directly:

```bash
python sft.py -c configs/qwen/train/lora/arc_fixed_k_max_lora.yaml --seed 42
python sft.py -c configs/qwen/train/full_ft/arc_fixed_k_max_full.yaml --seed 42
```

For Qwen and GPT-OSS full fine-tuning on two GPUs, launch with Accelerate and
enable FSDP:

```bash
accelerate launch --num_processes 2 sft.py -c configs/qwen/train/full_ft/arc_fixed_k_max_full.yaml --seed 42 --fsdp
accelerate launch --num_processes 2 sft.py -c configs/gpt_oss/train/full_ft/arc_fixed_k_max_full.yaml --seed 42 --fsdp
```

`./run_all.sh full_ft` automatically uses the Accelerate/FSDP launch for
`qwen` and `gpt_oss`, while `lfm2` and `olmoe` still run through the normal
single-process command. Override the GPU count with `FSDP_NUM_PROCESSES=...`.

A training run writes to:

```text
ckpt/<run_name>__seed_<seed>/
```

Final artifacts are saved as:

```text
final_adapter   # LoRA runs
final_model     # full_ft runs
```

Run the full training matrix for one fine-tuning mode:

```bash
./run_all.sh lora
./run_all.sh full_ft
```

For VMs with limited disk space, delete each completed run checkpoint before
starting the next run:

```bash
./run_all.sh lora --delete-checkpoints
./run_all.sh full_ft --delete-checkpoints
```

Timing logs are written to:

```text
logs/training_times_lora.tsv
logs/training_times_full_ft.tsv
```

## Evaluation

Run one eval config directly:

```bash
python eval.py -c configs/qwen/eval/lora/arc_fixed_k_max_lora.yaml
python eval.py -c configs/qwen/eval/full_ft/arc_fixed_k_max_full.yaml
```

`eval.py` also accepts overrides used by the batch runner:

```bash
python eval.py \
  -c configs/qwen/eval/lora/arc_fixed_k_max_lora.yaml \
  --seed 42 \
  --run-name qwen_arc_fixed_k_max_lora__seed_42 \
  --checkpoint-mode lora_adapter \
  --checkpoint-path ckpt/qwen_arc_fixed_k_max_lora__seed_42/final_adapter
```

Run the full evaluation matrix:

```bash
./eval_all.sh lora
./eval_all.sh full_ft
```

Evaluation results are written under:

```text
eval_results/<benchmark>/<run_name>/
├── metrics.json
└── predictions.jsonl
```

The current analysis workflow expects seeded result directories, for example:

```text
eval_results/arc/qwen_arc_fixed_k_max_lora__seed_42/
```

## Routing Behavior

The curriculum code controls MoE top-k routing during training. The routing
method comes from each training config:

- `fixed_k_max`: keep the model's default maximum top-k
- `fixed_k_1`: force top-k to 1
- `linear_k_1_to_topk`: linearly increase from 1 to the default top-k
- `frontloaded`, `backloaded`, `linear_mid_start`, `warmup`, `jump_warmup`:
  schedule variants for when and how top-k changes during training

During evaluation, `fixed_k_1` runs also force top-k to 1. Full-model
generation temporarily disables `output_router_logits` while calling
`model.generate()` to avoid router-logit output overhead during generation.


## Operational Notes

- Training and evaluation are config-driven. To add a new experiment, add the
  corresponding YAMLs and extend the model, task, or routing arrays in
  `run_all.sh` and `eval_all.sh`.
- Batch scripts run sequentially by design. This keeps GPU memory usage
  predictable and makes runs easier to resume or inspect.
- The seeded checkpoint and result naming convention is part of the analysis
  contract. Keep the `__seed_<seed>` suffix for runs that should appear in the
  notebook.
- Use `--delete-checkpoints` only when evaluation will be run later from a
  different checkpoint source or when checkpoints are intentionally disposable.
