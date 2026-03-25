def set_top_k(model, k: int):
    for _, module in model.named_modules():
        for attr in dir(module):
            if "top_k" in attr.lower():
                try:
                    val = getattr(module, attr)
                    if isinstance(val, int):
                        setattr(module, attr, k)
                except Exception:
                    pass
