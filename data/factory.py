from data.gsm8k import GSM8KTaskAdapter
from data.arc import ARCTaskAdapter
from data.sciq import SciQTaskAdapter


TASK_REGISTRY = {
    "openai/gsm8k": GSM8KTaskAdapter,
    "allenai/ai2_arc": ARCTaskAdapter,
    "allenai/sciq": SciQTaskAdapter,
}


def get_task_adapter(cfg, tokenizer):
    if cfg.dataset_id not in TASK_REGISTRY:
        raise ValueError(f"Unsupported dataset_id: {cfg.dataset_id}")
    return TASK_REGISTRY[cfg.dataset_id](cfg, tokenizer)
