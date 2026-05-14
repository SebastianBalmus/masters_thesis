import yaml
import wandb
import argparse
import gc
import os
from dotmap import DotMap

import torch
from trl import SFTConfig
from training.trainer import CurriculumSFTTrainer
from training.metrics_callback import (
    GenerativeEvalCallback,
    RoutingMetricsLoggingCallback,
    ValidationTrackingCallback,
)
from training.moe_metrics import MinimalMoeMetricCollector

from core.seed import set_seed
from training.pipeline import build_pipeline
from curriculum.callbacks import (
    FixedMoeRoutingCallback,
    ScheduledMoeRoutingCallback,
    build_stage_to_topk,
)
from evaluation.benchmarks.factory import get_benchmark_adapter
from evaluation.model_loader import load_model_for_eval, load_tokenizer_for_eval
from evaluation.runner import run_evaluation

ROUTING_METHOD_FIXED_MAX = "fixed_k_max"
ROUTING_METHOD_FIXED_ONE = "fixed_k_1"
ROUTING_METHOD_LINEAR = "linear_k_1_to_topk"
ROUTING_METHOD_WARMUP = "warmup"
ROUTING_METHOD_LINEAR_MID_START = "linear_mid_start"
ROUTING_METHOD_FRONTLOADED = "frontloaded"
ROUTING_METHOD_BACKLOADED = "backloaded"
ROUTING_METHOD_JUMP_WARMUP = "jump_warmup"

DEFAULT_FSDP_TRANSFORMER_LAYERS = {
    "qwen": "Qwen2MoeDecoderLayer",
    "gpt-oss": "GptOssDecoderLayer",
}


def seed_suffix(seed: int) -> str:
    return f"__seed_{int(seed)}"


def is_global_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def is_distributed_process() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def resolve_fsdp_enabled(cfg) -> bool:
    if "fsdp" in cfg:
        return bool(cfg.fsdp)
    fsdp_config = cfg.get("fsdp_config", None)
    if fsdp_config is not None and "enabled" in fsdp_config:
        return bool(fsdp_config.enabled)
    return False


def resolve_fsdp_transformer_layer_cls(cfg, model=None) -> str:
    fsdp_config = cfg.get("fsdp_config", None)
    if fsdp_config is not None and fsdp_config.get("transformer_layer_cls_to_wrap"):
        return str(fsdp_config.transformer_layer_cls_to_wrap)

    no_split_modules = getattr(model, "_no_split_modules", None)
    if no_split_modules:
        return str(no_split_modules[0])

    model_id = str(cfg.model_id).lower()
    for key, layer_cls in DEFAULT_FSDP_TRANSFORMER_LAYERS.items():
        if key in model_id:
            return layer_cls

    raise ValueError(
        "FSDP requires fsdp_config.transformer_layer_cls_to_wrap for this model."
    )


def build_fsdp_training_args(cfg, model=None) -> dict:
    if not resolve_fsdp_enabled(cfg):
        return {}

    if cfg.get("use_lora", False):
        raise ValueError("FSDP is only supported for full fine-tuning in this script.")
    if not bool(cfg.get("use_full", False)):
        raise ValueError("Set use_full: true when enabling FSDP.")

    transformer_layer_cls = resolve_fsdp_transformer_layer_cls(cfg, model=model)
    return {
        "fsdp": "full_shard auto_wrap",
        "fsdp_config": {
            "transformer_layer_cls_to_wrap": transformer_layer_cls,
            "activation_checkpointing": bool(
                cfg.get("fsdp_activation_checkpointing", True)
            ),
            "state_dict_type": "FULL_STATE_DICT",
            "use_orig_params": bool(cfg.get("fsdp_use_orig_params", True)),
        },
    }


def resolve_routing_method(cfg):
    return str(cfg.get("routing_method", ROUTING_METHOD_FIXED_MAX))


def resolve_routing_transition_ratio(cfg, routing_method: str) -> float:
    ratio = float(cfg.get("routing_transition_ratio", 0.1))
    if routing_method in {ROUTING_METHOD_WARMUP, ROUTING_METHOD_JUMP_WARMUP}:
        if not (0.0 < ratio <= 1.0):
            raise ValueError(
                f"routing_transition_ratio must be in (0, 1] for {routing_method}, got {ratio}"
            )
    return ratio


def build_dynamic_routing_callback(cfg, pipeline, routing_method: str):
    if not pipeline["supports_moe"] or pipeline["default_topk"] is None:
        raise ValueError(
            f"routing_method={routing_method} but the model config does not appear to support MoE top-k routing."
        )

    default_topk = int(pipeline["default_topk"])
    max_steps = int(pipeline["max_steps"])
    transition_ratio = resolve_routing_transition_ratio(cfg, routing_method)

    if routing_method == ROUTING_METHOD_LINEAR:
        stage_to_topk = build_stage_to_topk(
            num_stages=default_topk,
            target_topk=default_topk,
            start_topk=1,
        )
        return ScheduledMoeRoutingCallback(
            total_steps=max_steps,
            stage_to_topk=stage_to_topk,
            method_name=routing_method,
        )

    if routing_method == ROUTING_METHOD_LINEAR_MID_START:
        start_topk = max(1, default_topk // 2)
        num_stages = default_topk - start_topk + 1
        stage_to_topk = build_stage_to_topk(
            num_stages=num_stages,
            target_topk=default_topk,
            start_topk=start_topk,
        )
        return ScheduledMoeRoutingCallback(
            total_steps=max_steps,
            stage_to_topk=stage_to_topk,
            method_name=routing_method,
        )

    if routing_method == ROUTING_METHOD_WARMUP:
        stage_to_topk = build_stage_to_topk(
            num_stages=default_topk,
            target_topk=default_topk,
            start_topk=1,
        )
        return ScheduledMoeRoutingCallback(
            total_steps=max_steps,
            stage_to_topk=stage_to_topk,
            method_name=routing_method,
            transition_ratio=transition_ratio,
            post_transition_topk=default_topk,
        )

    if routing_method == ROUTING_METHOD_FRONTLOADED:
        stage_to_topk = build_stage_to_topk(
            num_stages=default_topk,
            target_topk=default_topk,
            start_topk=1,
        )
        return ScheduledMoeRoutingCallback(
            total_steps=max_steps,
            stage_to_topk=stage_to_topk,
            method_name=routing_method,
            stage_weights=list(range(1, default_topk + 1)),
        )

    if routing_method == ROUTING_METHOD_BACKLOADED:
        stage_to_topk = build_stage_to_topk(
            num_stages=default_topk,
            target_topk=default_topk,
            start_topk=1,
        )
        return ScheduledMoeRoutingCallback(
            total_steps=max_steps,
            stage_to_topk=stage_to_topk,
            method_name=routing_method,
            stage_weights=list(range(default_topk, 0, -1)),
        )

    if routing_method == ROUTING_METHOD_JUMP_WARMUP:
        return ScheduledMoeRoutingCallback(
            total_steps=max_steps,
            stage_to_topk={0: 1},
            method_name=routing_method,
            transition_ratio=transition_ratio,
            post_transition_topk=default_topk,
        )

    raise ValueError(f"Unsupported dynamic routing_method: {routing_method}")


def dataset_id_to_benchmark_name(dataset_id: str) -> str:
    mapping = {
        "openai/gsm8k": "gsm8k",
        "allenai/ai2_arc": "arc",
        "allenai/sciq": "sciq",
    }
    if dataset_id not in mapping:
        raise ValueError(f"Unsupported dataset_id for evaluation: {dataset_id}")
    return mapping[dataset_id]


def build_post_train_eval_cfg(
    cfg, checkpoint_mode: str, checkpoint_path: str, run_name: str
):
    return DotMap(
        {
            "seed": cfg.seed,
            "model_id": cfg.model_id,
            "run_name": run_name,
            "routing_method": resolve_routing_method(cfg),
            "checkpoint": {
                "mode": checkpoint_mode,
                "path": checkpoint_path,
            },
            "eval": {
                "benchmark": dataset_id_to_benchmark_name(cfg.dataset_id),
                "results_dir": cfg.get("results_dir", "eval_results"),
                "batch_size": int(
                    cfg.get(
                        "post_train_eval_batch_size", cfg.per_device_eval_batch_size
                    )
                ),
                "max_new_tokens": int(cfg.max_new_tokens),
                "split": "test",
                "limit": None,
                "few_shot": False,
                "temperature": 0.0,
                "do_sample": False,
                "sort_by_length": bool(cfg.get("sort_eval_by_length", True)),
                "compile_model": bool(cfg.get("compile_eval_model", False)),
                "attn_implementation": cfg.get("eval_attn_implementation", "sdpa"),
            },
        }
    )


def run_checkpoint_test_eval(
    cfg, checkpoint_mode: str, checkpoint_path: str, run_name: str
):
    eval_cfg = build_post_train_eval_cfg(
        cfg=cfg,
        checkpoint_mode=checkpoint_mode,
        checkpoint_path=checkpoint_path,
        run_name=run_name,
    )
    tokenizer = load_tokenizer_for_eval(eval_cfg)
    model, load_info = load_model_for_eval(eval_cfg)
    benchmark = get_benchmark_adapter(eval_cfg, tokenizer)

    metrics = run_evaluation(
        cfg=eval_cfg,
        model=model,
        tokenizer=tokenizer,
        benchmark=benchmark,
        load_info=load_info,
    )

    del model
    del tokenizer
    del benchmark
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def release_training_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def extract_accuracy(metrics: dict, metric_key: str):
    if metric_key not in metrics:
        raise KeyError(
            f"Expected metric {metric_key!r} in metrics: {list(metrics.keys())}"
        )
    return float(metrics[metric_key])


def main(cfg):
    cfg = DotMap(cfg)
    suffix = seed_suffix(cfg.seed)

    cfg.output_dir = f"{cfg.output_dir}{suffix}"
    if "run_name" in cfg and cfg.run_name is not None:
        cfg.run_name = f"{cfg.run_name}{suffix}"
    if (
        "wandb_config" in cfg
        and cfg.wandb_config is not None
        and "name" in cfg.wandb_config
    ):
        cfg.wandb_config.name = f"{cfg.wandb_config.name}{suffix}"

    if cfg.report_to == "wandb" and is_global_main_process():
        wandb.init(
            project=cfg.wandb_config.project,
            name=cfg.wandb_config.name,
            group=cfg.wandb_config.group,
            config=cfg.toDict(),
        )

    set_seed(cfg.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    pipeline = build_pipeline(cfg)

    callbacks = []
    routing_method = resolve_routing_method(cfg)
    routing_callback = None
    moe_metrics_collector = None

    if routing_method in {
        ROUTING_METHOD_LINEAR,
        ROUTING_METHOD_WARMUP,
        ROUTING_METHOD_LINEAR_MID_START,
        ROUTING_METHOD_FRONTLOADED,
        ROUTING_METHOD_BACKLOADED,
        ROUTING_METHOD_JUMP_WARMUP,
    }:
        routing_callback = build_dynamic_routing_callback(
            cfg=cfg,
            pipeline=pipeline,
            routing_method=routing_method,
        )
    elif routing_method == ROUTING_METHOD_FIXED_ONE:
        if not pipeline["supports_moe"]:
            raise ValueError(
                "routing_method=fixed_k_1 but the model config does not appear to support MoE top-k routing."
            )
        routing_callback = FixedMoeRoutingCallback(
            topk=1,
            method_name=routing_method,
        )
    elif routing_method == ROUTING_METHOD_FIXED_MAX:
        if pipeline["supports_moe"] and pipeline["default_topk"] is not None:
            routing_callback = FixedMoeRoutingCallback(
                topk=pipeline["default_topk"],
                method_name=routing_method,
            )
    else:
        raise ValueError(f"Unsupported routing_method: {routing_method}")

    if routing_callback is not None:
        callbacks.append(routing_callback)

    enable_task_metrics = bool(cfg.get("enable_task_metrics_during_training", False))
    task_metric_eval_steps = cfg.get("task_metric_eval_steps", cfg.eval_steps)
    generative_eval_callback = None

    if enable_task_metrics and pipeline["task_adapter"].has_task_metrics():
        generative_eval_callback = GenerativeEvalCallback(
            task_adapter=pipeline["task_adapter"],
            tokenizer=pipeline["tokenizer"],
            raw_eval_dataset=pipeline["raw_val_ds"],
            batch_size=cfg.per_device_eval_batch_size,
            max_new_tokens=int(cfg.get("eval_max_new_tokens", cfg.max_new_tokens)),
            enabled=True,
            step_interval=task_metric_eval_steps,
        )
        callbacks.append(generative_eval_callback)
    elif not pipeline["task_adapter"].has_task_metrics():
        raise ValueError(
            "This training protocol requires task-specific validation accuracy."
        )

    metric_name = f"eval_{pipeline['task_adapter'].get_metric_key()}"
    if pipeline["supports_moe"] and pipeline["default_topk"] is not None:
        moe_metrics_collector = MinimalMoeMetricCollector(
            default_topk=pipeline["default_topk"],
            current_k_getter=(
                None
                if routing_callback is None
                else lambda: getattr(routing_callback, "current_topk", None)
            ),
        )

    callbacks.append(
        RoutingMetricsLoggingCallback(
            routing_callback=routing_callback,
            moe_metrics_collector=moe_metrics_collector,
        )
    )
    tracking_callback = ValidationTrackingCallback(
        metric_name=metric_name,
        output_dir=cfg.output_dir,
        routing_callback=routing_callback,
    )
    callbacks.append(tracking_callback)
    eval_strategy = "steps" if enable_task_metrics else "epoch"

    fsdp_training_args = build_fsdp_training_args(cfg, model=pipeline["model"])
    training_args = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=1.0,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=cfg.eval_steps,
        save_strategy="no",
        save_steps=cfg.eval_steps,
        report_to=cfg.report_to if is_global_main_process() else "none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        max_length=cfg.max_seq_length,
        packing=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        **fsdp_training_args,
    )

    trainer_kwargs = dict(
        model=pipeline["model"],
        args=training_args,
        train_dataset=pipeline["train_ds"],
        eval_dataset=pipeline["val_ds"],
        processing_class=pipeline["tokenizer"],
        data_collator=pipeline["data_collator"],
        callbacks=callbacks,
        moe_metrics_collector=moe_metrics_collector,
    )

    if pipeline["peft_config"] is not None:
        trainer_kwargs["peft_config"] = pipeline["peft_config"]

    trainer = CurriculumSFTTrainer(**trainer_kwargs)
    train_result = trainer.train()

    final_artifact_dir = (
        f"{cfg.output_dir}/final_adapter"
        if pipeline["peft_config"] is not None
        else f"{cfg.output_dir}/final_model"
    )
    trainer.save_model(final_artifact_dir)
    if trainer.is_world_process_zero():
        pipeline["tokenizer"].save_pretrained(final_artifact_dir)

    if is_distributed_process():
        trainer.accelerator.wait_for_everyone()

    if generative_eval_callback is not None:
        generative_eval_callback.force_run_next_eval()
    trainer.evaluate()

    final_step = int(trainer.state.global_step)
    training_runtime_seconds = train_result.metrics.get("training_runtime_seconds")
    validation_runtime_seconds = train_result.metrics.get("validation_runtime_seconds")
    task_metric_key = pipeline["task_adapter"].get_metric_key()
    is_peft_model = pipeline["peft_config"] is not None

    run_name_base = cfg.get("run_name", cfg.wandb_config.name)
    final_checkpoint_mode = "lora_adapter" if is_peft_model else "full_model"

    trainer = None
    trainer_kwargs = None
    training_args = None
    callbacks = None
    train_result = None
    generative_eval_callback = None
    moe_metrics_collector = None
    routing_callback = None
    pipeline = None
    release_training_memory()

    if not is_global_main_process():
        if wandb.run is not None:
            wandb.finish()
        return

    final_test_metrics = run_checkpoint_test_eval(
        cfg=cfg,
        checkpoint_mode=final_checkpoint_mode,
        checkpoint_path=final_artifact_dir,
        run_name=run_name_base,
    )
    test_accuracy_at_final = extract_accuracy(
        final_test_metrics,
        task_metric_key,
    )
    inference_runtime_seconds = final_test_metrics.get("inference_runtime_seconds")

    summary = tracking_callback.get_summary(
        test_accuracy_at_final=test_accuracy_at_final,
        training_runtime_seconds=training_runtime_seconds,
        validation_runtime_seconds=validation_runtime_seconds,
        inference_runtime_seconds=inference_runtime_seconds,
        final_step=final_step,
    )
    tracking_callback.write_summary(summary)

    if wandb.run is not None:
        for key, value in summary.items():
            if isinstance(value, (int, float)) or value is None:
                wandb.run.summary[key] = value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT Training Script")
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default="configs/sft_config.yaml",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the training seed from the config file.",
    )
    parser.add_argument(
        "--fsdp",
        action="store_true",
        help="Enable FSDP full fine-tuning for this run.",
    )
    parser.add_argument(
        "--no-fsdp",
        action="store_true",
        help="Disable FSDP even if the config enables it.",
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.fsdp and args.no_fsdp:
        raise ValueError("Use only one of --fsdp or --no-fsdp.")
    if args.fsdp:
        cfg["fsdp"] = True
    if args.no_fsdp:
        cfg["fsdp"] = False

    main(cfg)
