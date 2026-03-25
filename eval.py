import argparse
import yaml
from dotmap import DotMap

import torch

from core.seed import set_seed
from evaluation.benchmarks.factory import get_benchmark_adapter
from evaluation.model_loader import load_model_for_eval, load_tokenizer_for_eval
from evaluation.runner import run_evaluation

torch.set_float32_matmul_precision("high")


def main(cfg):
    cfg = DotMap(cfg)

    set_seed(cfg.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("This evaluation script currently expects CUDA.")

    tokenizer = load_tokenizer_for_eval(cfg)
    model, load_info = load_model_for_eval(cfg)
    benchmark = get_benchmark_adapter(cfg, tokenizer)

    run_evaluation(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        benchmark=benchmark,
        load_info=load_info,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        required=True,
        help="Path to eval YAML config",
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
