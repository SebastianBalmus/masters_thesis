import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig


def get_dtype():
    return (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )


def load_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_model(cfg):
    model_id = cfg.model_id
    dtype = get_dtype()
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    use_lora = bool(cfg.get("use_lora", False))

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    peft_config = None
    if use_lora:
        peft_config = LoraConfig(
            r=cfg.lora_config.r,
            lora_alpha=cfg.lora_config.lora_alpha,
            lora_dropout=cfg.lora_config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )

    model.config.use_cache = False
    return model, config, peft_config
