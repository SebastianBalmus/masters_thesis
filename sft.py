import os
import yaml
import argparse
from dotmap import DotMap

import torch
from trl import SFTConfig
from training.trainer import CurriculumSFTTrainer

from core.seed import set_seed
from training.pipeline import build_pipeline
from curriculum.callbacks import (
    StagewiseMoeCurriculumCallback,
    StandardModelRoutingCallback,
)


def main(cfg):
    cfg = DotMap(cfg)

    if cfg.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = cfg.wandb_config.project
        os.environ["WANDB_NAME"] = cfg.wandb_config.name

    set_seed(cfg.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    pipeline = build_pipeline(cfg)

    callbacks = []

    if cfg.use_model_curriculum:
        if not pipeline["supports_moe"]:
            raise ValueError(
                "use_model_curriculum=True but the model config does not appear to support MoE top-k routing."
            )
        if pipeline["difficulty_to_topk"] is None:
            raise ValueError(
                "use_model_curriculum=True but difficulty_to_topk could not be constructed."
            )

        callbacks.append(
            StagewiseMoeCurriculumCallback(
                curriculum_state=pipeline["curriculum_state"],
                difficulty_to_topk=pipeline["difficulty_to_topk"],
            )
        )
    else:
        if pipeline["supports_moe"] and pipeline["default_topk"] is not None:
            callbacks.append(
                StandardModelRoutingCallback(
                    default_topk=pipeline["default_topk"],
                )
            )

    training_args = SFTConfig(
        output_dir=cfg.output_dir,
        max_steps=pipeline["max_steps"],
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
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
    )

    if pipeline["peft_config"] is not None:
        trainer_kwargs["peft_config"] = pipeline["peft_config"]

    trainer = CurriculumSFTTrainer(**trainer_kwargs)
    trainer.train()

    trainer.model.save_pretrained(f"{cfg.output_dir}/final_adapter")
    pipeline["tokenizer"].save_pretrained(f"{cfg.output_dir}/final_adapter")


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
