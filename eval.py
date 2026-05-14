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


def apply_cli_overrides(cfg, args):
    cfg.setdefault("seed", 42)

    if args.seed is not None:
        cfg["seed"] = args.seed

    if args.run_name is not None:
        cfg["run_name"] = args.run_name

    if args.checkpoint_mode is not None or args.checkpoint_path is not None:
        cfg.setdefault("checkpoint", {})

    if args.checkpoint_mode is not None:
        cfg["checkpoint"]["mode"] = args.checkpoint_mode

    if args.checkpoint_path is not None:
        cfg["checkpoint"]["path"] = args.checkpoint_path

    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        required=True,
        help="Path to eval YAML config",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override eval seed")
    parser.add_argument("--run-name", type=str, default=None, help="Override eval run name")
    parser.add_argument(
        "--checkpoint-mode",
        type=str,
        choices=["base", "lora_adapter", "full_model"],
        default=None,
        help="Override checkpoint.mode",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Override checkpoint.path",
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg = apply_cli_overrides(cfg, args)
    main(cfg)
