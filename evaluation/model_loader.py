from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.moe import set_top_k

torch.set_float32_matmul_precision("high")


def should_force_eval_topk_one(cfg) -> bool:
    routing_method = getattr(cfg, "routing_method", None)
    if routing_method is not None:
        return str(routing_method) == "fixed_k_1"

    run_name = str(getattr(cfg, "run_name", "") or "")
    checkpoint_path = str(getattr(getattr(cfg, "checkpoint", None), "path", "") or "")
    return "fixed_k_1" in run_name or "fixed_k_1" in checkpoint_path


def maybe_apply_eval_topk_override(model, cfg):
    if should_force_eval_topk_one(cfg):
        set_top_k(model, k=1)
        return 1
    return None


def get_torch_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def maybe_compile_model(model, enabled: bool):
    if not enabled:
        return model

    if not hasattr(torch, "compile"):
        print("torch.compile not available; skipping.")
        return model

    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        print("Model compiled with torch.compile.")
        return model
    except Exception as e:
        print(f"torch.compile failed; continuing without compilation. Error: {e}")
        return model


def load_tokenizer_with_fallback(primary_path: str, fallback_path: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(primary_path, trust_remote_code=True)

    if tokenizer.pad_token is None and fallback_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            fallback_path,
            trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    return tokenizer


def load_base_model(model_name_or_path: str, attn_implementation: str):
    kwargs = dict(
        torch_dtype=get_torch_dtype(),
        trust_remote_code=True,
    )

    if torch.cuda.is_available():
        kwargs["device_map"] = "cuda:0"

    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            attn_implementation=attn_implementation,
            **kwargs,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    except Exception as e:
        print(
            f"Warning: failed to load with attn_implementation={attn_implementation!r}: {e}"
        )
        print("Falling back to default attention implementation.")
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)


def load_tokenizer_for_eval(cfg):
    mode = cfg.checkpoint.mode
    model_id = cfg.model_id
    ckpt_path = cfg.checkpoint.path

    if mode == "base":
        return load_tokenizer_with_fallback(model_id)

    if mode == "lora_adapter":
        return load_tokenizer_with_fallback(
            primary_path=ckpt_path,
            fallback_path=model_id,
        )

    if mode == "full_model":
        return load_tokenizer_with_fallback(primary_path=ckpt_path)

    raise ValueError(f"Unsupported checkpoint.mode: {mode}")


def load_model_for_eval(cfg):
    mode = cfg.checkpoint.mode
    model_id = cfg.model_id
    ckpt_path = cfg.checkpoint.path
    attn_implementation = cfg.eval.attn_implementation
    compile_model = cfg.eval.compile_model

    if mode == "base":
        model = load_base_model(model_id, attn_implementation=attn_implementation)
        forced_topk = maybe_apply_eval_topk_override(model, cfg)
        model.eval()
        model = maybe_compile_model(model, compile_model)

        return model, {
            "checkpoint_type": "base_model",
            "base_model_name_or_path": model_id,
            "checkpoint_path": None,
            "peft_merged": False,
            "forced_topk": forced_topk,
        }

    if mode == "lora_adapter":
        base_model = load_base_model(
            model_id,
            attn_implementation=attn_implementation,
        )
        model = PeftModel.from_pretrained(base_model, ckpt_path)

        merged = False
        try:
            model = model.merge_and_unload()
            merged = True
        except Exception as e:
            print(f"Warning: merge_and_unload failed; using adapter as-is. Error: {e}")

        forced_topk = maybe_apply_eval_topk_override(model, cfg)
        model.eval()
        model = maybe_compile_model(model, compile_model)

        return model, {
            "checkpoint_type": "lora_adapter",
            "base_model_name_or_path": model_id,
            "checkpoint_path": ckpt_path,
            "peft_merged": merged,
            "forced_topk": forced_topk,
        }

    if mode == "full_model":
        model = load_base_model(
            ckpt_path,
            attn_implementation=attn_implementation,
        )
        forced_topk = maybe_apply_eval_topk_override(model, cfg)
        model.eval()
        model = maybe_compile_model(model, compile_model)

        return model, {
            "checkpoint_type": "full_model",
            "base_model_name_or_path": ckpt_path,
            "checkpoint_path": ckpt_path,
            "peft_merged": False,
            "forced_topk": forced_topk,
        }

    raise ValueError(f"Unsupported checkpoint.mode: {mode}")
