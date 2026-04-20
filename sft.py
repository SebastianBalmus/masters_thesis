import yaml
import wandb
import argparse
import gc
from pathlib import Path
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


def build_post_train_eval_cfg(cfg, checkpoint_mode: str, checkpoint_path: str, run_name: str):
    return DotMap(
        {
            "seed": cfg.seed,
            "model_id": cfg.model_id,
            "run_name": run_name,
            "checkpoint": {
                "mode": checkpoint_mode,
                "path": checkpoint_path,
            },
            "eval": {
                "benchmark": dataset_id_to_benchmark_name(cfg.dataset_id),
                "results_dir": cfg.get("results_dir", "eval_results"),
                "batch_size": int(
                    cfg.get("post_train_eval_batch_size", cfg.per_device_eval_batch_size)
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


def run_checkpoint_test_eval(cfg, checkpoint_mode: str, checkpoint_path: str, run_name: str):
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
        raise KeyError(f"Expected metric {metric_key!r} in metrics: {list(metrics.keys())}")
    return float(metrics[metric_key])


def main(cfg):
    cfg = DotMap(cfg)

    if cfg.report_to == "wandb":
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

    enable_task_metrics = bool(cfg.get("enable_task_metrics_during_training", True))
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
        raise ValueError("This training protocol requires task-specific validation accuracy.")

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
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.eval_steps,
        report_to=cfg.report_to,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        max_length=cfg.max_seq_length,
        packing=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
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
    trainer.model.save_pretrained(final_artifact_dir)
    pipeline["tokenizer"].save_pretrained(final_artifact_dir)

    if generative_eval_callback is not None:
        generative_eval_callback.force_run_next_eval()
    trainer.evaluate()

    final_step = int(trainer.state.global_step)
    train_runtime = train_result.metrics.get("train_runtime")
    best_step = tracking_callback.best_step
    task_metric_key = pipeline["task_adapter"].get_metric_key()
    is_peft_model = pipeline["peft_config"] is not None

    run_name_base = cfg.get("run_name", cfg.wandb_config.name)
    final_run_name = f"{run_name_base}__final"
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

    final_test_metrics = run_checkpoint_test_eval(
        cfg=cfg,
        checkpoint_mode=final_checkpoint_mode,
        checkpoint_path=final_artifact_dir,
        run_name=final_run_name,
    )
    test_accuracy_at_final = extract_accuracy(
        final_test_metrics,
        task_metric_key,
    )

    best_checkpoint_dir = None
    if best_step is not None:
        candidate = f"{cfg.output_dir}/checkpoint-{best_step}"
        best_checkpoint_dir = candidate if Path(candidate).is_dir() else None

        if best_checkpoint_dir is None and best_step == final_step:
            best_checkpoint_dir = final_artifact_dir

    test_accuracy_at_best = None
    if best_checkpoint_dir is not None:
        best_run_name = f"{run_name_base}__best"
        best_checkpoint_mode = "lora_adapter" if is_peft_model else "full_model"
        best_test_metrics = run_checkpoint_test_eval(
            cfg=cfg,
            checkpoint_mode=best_checkpoint_mode,
            checkpoint_path=best_checkpoint_dir,
            run_name=best_run_name,
        )
        test_accuracy_at_best = extract_accuracy(
            best_test_metrics,
            task_metric_key,
        )

    summary = tracking_callback.get_summary(
        test_accuracy_at_best=test_accuracy_at_best,
        test_accuracy_at_final=test_accuracy_at_final,
        train_runtime=train_runtime,
        final_step=final_step,
    )
    tracking_callback.write_summary(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT Training Script")
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        default="configs/sft_config.yaml",
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
