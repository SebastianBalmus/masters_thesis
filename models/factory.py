import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training


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

    peft_config = None

    if cfg.use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            quantization_config=bnb_config,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=cfg.lora_config.r,
            lora_alpha=cfg.lora_config.lora_alpha,
            lora_dropout=cfg.lora_config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

    model.config.use_cache = False
    return model, config, peft_config
