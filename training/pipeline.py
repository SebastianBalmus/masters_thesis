import math
import os

from data.factory import get_task_adapter
from models.factory import build_model, load_tokenizer
from models.capabilities import supports_moe_topk
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

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_batch_size = (
        cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * world_size
    )
    max_steps = math.ceil(len(train_ds) / effective_batch_size)

    model, model_config, peft_config = build_model(cfg)

    data_collator = CausalLMCollator(tokenizer)

    moe_supported = supports_moe_topk(model_config)

    default_topk = None

    if moe_supported and hasattr(model_config, "num_experts_per_tok"):
        default_topk = int(model_config.num_experts_per_tok)

    return {
        "model": model,
        "model_config": model_config,
        "peft_config": peft_config,
        "tokenizer": tokenizer,
        "task_adapter": task,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "raw_val_ds": raw_val_ds,
        "max_steps": max_steps,
        "data_collator": data_collator,
        "supports_moe": moe_supported,
        "default_topk": default_topk,
    }
