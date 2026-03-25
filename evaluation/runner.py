import json
from pathlib import Path

import torch
from tqdm import tqdm

from evaluation.utils import batchify, num_batches

from pathlib import Path


def build_eval_run_name(cfg) -> str:
    custom_name = getattr(cfg, "run_name", None)
    if custom_name:
        return str(custom_name)

    if cfg.checkpoint.mode == "base":
        model_name = Path(str(cfg.model_id)).name
        return f"base__{model_name}"

    ckpt_name = Path(str(cfg.checkpoint.path)).name

    if cfg.checkpoint.mode == "lora_adapter":
        return f"finetune_lora__{ckpt_name}"

    if cfg.checkpoint.mode == "full_model":
        return f"finetune_full__{ckpt_name}"

    raise ValueError(f"Unsupported checkpoint.mode: {cfg.checkpoint.mode}")


def pin_and_move_to_device(batch_tensors, device):
    moved = {}
    for key, value in batch_tensors.items():
        if value.device.type == "cpu" and torch.cuda.is_available():
            value = value.pin_memory()
        moved[key] = (
            value.to(device, non_blocking=True) if hasattr(value, "to") else value
        )
    return moved


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts, cfg):
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        pad_to_multiple_of=8,
    )
    enc = pin_and_move_to_device(enc, model.device)

    generate_kwargs = dict(
        **enc,
        max_new_tokens=cfg.eval.max_new_tokens,
        do_sample=cfg.eval.do_sample,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if cfg.eval.do_sample:
        generate_kwargs["temperature"] = cfg.eval.temperature
    else:
        generate_kwargs["num_beams"] = 1

    try:
        outputs = model.generate(cache_implementation="static", **generate_kwargs)
    except Exception:
        outputs = model.generate(**generate_kwargs)

    prompt_len = enc["input_ids"].shape[1]
    gen_only = outputs[:, prompt_len:]
    return tokenizer.batch_decode(gen_only, skip_special_tokens=True)


def maybe_sort_examples_by_length(examples, benchmark, tokenizer, enabled: bool):
    if not enabled:
        return examples

    print("Sorting examples by tokenized prompt length...")
    return sorted(
        examples,
        key=lambda ex: len(tokenizer(benchmark.format_prompt(ex)).input_ids),
    )


def run_evaluation(cfg, model, tokenizer, benchmark, load_info):
    dataset = benchmark.load_split()

    if cfg.eval.limit is not None:
        dataset = dataset.select(range(min(cfg.eval.limit, len(dataset))))

    examples = list(dataset)
    examples = maybe_sort_examples_by_length(
        examples, benchmark, tokenizer, cfg.eval.sort_by_length
    )

    run_name = build_eval_run_name(cfg)

    out_dir = Path(cfg.eval.results_dir) / cfg.eval.benchmark / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = out_dir / "predictions.jsonl"
    metrics_path = out_dir / "metrics.json"

    rows = []

    with predictions_path.open("w", encoding="utf-8") as pred_f:
        for batch_examples in tqdm(
            batchify(examples, cfg.eval.batch_size),
            total=num_batches(len(examples), cfg.eval.batch_size),
            desc=f"Evaluating {cfg.eval.benchmark}",
        ):
            prompts = [benchmark.format_prompt(ex) for ex in batch_examples]
            generations = generate_batch(model, tokenizer, prompts, cfg)

            for ex, prompt, gen_text in zip(batch_examples, prompts, generations):
                gold = benchmark.extract_gold_answer(ex)
                pred = benchmark.extract_predicted_answer(gen_text)
                is_correct = pred == gold

                row = {
                    "prompt": prompt,
                    "generated_text": gen_text,
                    "predicted_answer_extracted": pred,
                    "gold_answer_extracted": gold,
                    "correct": is_correct,
                }

                # Preserve benchmark-specific raw fields when useful
                if "question" in ex:
                    row["question"] = ex["question"]
                if "answer" in ex:
                    row["gold_answer_raw"] = ex["answer"]
                if "choices" in ex:
                    row["choices"] = ex["choices"]
                if "answerKey" in ex:
                    row["answer_key_raw"] = ex["answerKey"]
                if "correct_answer" in ex:
                    row["correct_answer_text"] = ex["correct_answer"]
                if "distractor1" in ex:
                    row["distractor1"] = ex["distractor1"]
                if "distractor2" in ex:
                    row["distractor2"] = ex["distractor2"]
                if "distractor3" in ex:
                    row["distractor3"] = ex["distractor3"]
                if "support" in ex:
                    row["support"] = ex["support"]

                rows.append(row)
                pred_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = benchmark.compute_metrics(rows)
    metrics.update(
        {
            "benchmark": cfg.eval.benchmark,
            "split": cfg.eval.split,
            "batch_size": cfg.eval.batch_size,
            "max_new_tokens": cfg.eval.max_new_tokens,
            "checkpoint_mode": cfg.checkpoint.mode,
            "checkpoint_path": cfg.checkpoint.path,
            "model_id": cfg.model_id,
            "predictions_path": str(predictions_path),
            **load_info,
        }
    )

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    return metrics
