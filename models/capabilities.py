def supports_moe_topk(config) -> bool:
    return hasattr(config, "num_experts_per_tok")
