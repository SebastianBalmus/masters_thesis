import math

from data.factory import get_task_adapter
from models.factory import build_model, load_tokenizer
from models.capabilities import supports_moe_topk
from curriculum.state import CurriculumState
from curriculum.iterable import RandomIterableDataset, CurriculumIterableDataset
from curriculum.callbacks import build_difficulty_to_topk
from training.collator import CausalLMCollator


def build_pipeline(cfg):
    tokenizer = load_tokenizer(
        cfg.model_id,
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
    )
    task = get_task_adapter(cfg, tokenizer)

    splits = task.build_splits()

    train_ds = splits["train_tokenized"]
    val_ds = splits["validation_tokenized"]
    raw_val_ds = splits["validation_raw"]

    effective_batch_size = (
        cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    )
    max_steps = math.ceil(len(train_ds) / effective_batch_size)

    model, model_config, peft_config = build_model(cfg)

    num_difficulty_levels = len(train_ds.features["difficulty"].names)

    curriculum_state = None
    if cfg.use_data_curriculum or cfg.use_model_curriculum:
        curriculum_state = CurriculumState(
            total_steps=max_steps,
            num_difficulty_levels=num_difficulty_levels,
        )

    if cfg.use_data_curriculum:
        train_dataset_for_trainer = CurriculumIterableDataset(
            dataset=train_ds,
            curriculum_state=curriculum_state,
            seed=cfg.seed,
        )
    else:
        train_dataset_for_trainer = RandomIterableDataset(
            dataset=train_ds,
            seed=cfg.seed,
        )

    data_collator = CausalLMCollator(tokenizer)

    moe_supported = supports_moe_topk(model_config)

    difficulty_to_topk = None
    default_topk = None

    if moe_supported and hasattr(model_config, "num_experts_per_tok"):
        default_topk = int(model_config.num_experts_per_tok)
        difficulty_to_topk = build_difficulty_to_topk(
            num_levels=num_difficulty_levels,
            target_topk=default_topk,
        )

    return {
        "model": model,
        "model_config": model_config,
        "peft_config": peft_config,
        "tokenizer": tokenizer,
        "task_adapter": task,
        "train_ds": train_dataset_for_trainer,
        "val_ds": val_ds,
        "raw_val_ds": raw_val_ds,
        "curriculum_state": curriculum_state,
        "num_difficulty_levels": num_difficulty_levels,
        "max_steps": max_steps,
        "data_collator": data_collator,
        "supports_moe": moe_supported,
        "difficulty_to_topk": difficulty_to_topk,
        "default_topk": default_topk,
    }
