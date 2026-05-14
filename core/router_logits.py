from contextlib import contextmanager


@contextmanager
def temporarily_disable_router_logits(model):
    configs = []
    seen = set()

    for candidate in (
        getattr(model, "config", None),
        getattr(getattr(model, "model", None), "config", None),
        getattr(getattr(model, "base_model", None), "config", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "config", None),
        getattr(model, "generation_config", None),
    ):
        if candidate is None:
            continue

        ident = id(candidate)
        if ident in seen:
            continue
        seen.add(ident)

        if hasattr(candidate, "output_router_logits"):
            configs.append((candidate, candidate.output_router_logits))
            candidate.output_router_logits = False

    try:
        yield
    finally:
        for config, old_value in configs:
            config.output_router_logits = old_value
