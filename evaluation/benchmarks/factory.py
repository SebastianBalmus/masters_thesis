from evaluation.benchmarks.gsm8k import GSM8KBenchmarkAdapter
from evaluation.benchmarks.arc import ARCBenchmarkAdapter
from evaluation.benchmarks.sciq import SciQBenchmarkAdapter


BENCHMARK_REGISTRY = {
    "gsm8k": GSM8KBenchmarkAdapter,
    "arc": ARCBenchmarkAdapter,
    "sciq": SciQBenchmarkAdapter,
}


def get_benchmark_adapter(cfg, tokenizer):
    benchmark_name = cfg.eval.benchmark
    if benchmark_name not in BENCHMARK_REGISTRY:
        raise ValueError(f"Unsupported benchmark: {benchmark_name}")
    return BENCHMARK_REGISTRY[benchmark_name](cfg, tokenizer)
