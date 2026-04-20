import json
from pathlib import Path
from contextlib import contextmanager

import torch
from tqdm import tqdm
from transformers import TrainerCallback


def batchify(items, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def append_exact_eval_log(state, log_dict: dict):
    step = int(state.global_step)
    merged = dict(log_dict)
    merged["step"] = step

    epoch = getattr(state, "epoch", None)
    if epoch is not None:
        try:
            merged["epoch"] = float(epoch)
        except Exception:
            pass

    state.log_history.append(merged)


def wandb_log_monotonic(payload: dict, step: int):
    try:
        import wandb

        if wandb.run is None or not payload:
            return

        current_step = getattr(wandb.run, "step", None)
        safe_step = step
        if isinstance(current_step, int):
            safe_step = max(step, current_step)

        wandb.log(payload, step=safe_step)
    except Exception as e:
        print(f"Warning: failed to log metrics to wandb: {e}")


def pin_and_move_to_device(
    batch_tensors: dict[str, torch.Tensor], device: torch.device
):
    moved = {}
    for key, value in batch_tensors.items():
        if value.device.type == "cpu":
            value = value.pin_memory()
        moved[key] = value.to(device, non_blocking=True)
    return moved


@contextmanager
def temporarily_disable_router_logits(model):
    configs = []
    seen = set()

    for candidate in (
        getattr(model, "config", None),
        getattr(getattr(model, "model", None), "config", None),
        getattr(getattr(model, "base_model", None), "config", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "config", None),
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


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts, max_new_tokens: int):
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        pad_to_multiple_of=8,
    )
    enc = pin_and_move_to_device(enc, model.device)

    gen_kwargs = dict(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    with temporarily_disable_router_logits(model):
        outputs = model.generate(**gen_kwargs)

    prompt_len = enc["input_ids"].shape[1]
    gen_only = outputs[:, prompt_len:]
    decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

    tokenizer.padding_side = original_padding_side
    return decoded


class GenerativeEvalCallback(TrainerCallback):
    def __init__(
        self,
        task_adapter,
        tokenizer,
        raw_eval_dataset,
        batch_size: int,
        max_new_tokens: int,
        enabled: bool = True,
        step_interval: int | None = None,
    ):
        self.task_adapter = task_adapter
        self.tokenizer = tokenizer
        self.raw_eval_dataset = raw_eval_dataset
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.enabled = bool(enabled)
        self.step_interval = None if step_interval is None else int(step_interval)
        self.force_next_eval = False

    def force_run_next_eval(self):
        self.force_next_eval = True

    def _should_run(self, step: int) -> bool:
        if self.force_next_eval:
            self.force_next_eval = False
            return True
        if not self.enabled:
            return False
        if self.step_interval is None or self.step_interval <= 0:
            return True
        return step > 0 and step % self.step_interval == 0

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if model is None:
            return control

        if not self.task_adapter.has_task_metrics():
            return control

        if not self._should_run(int(state.global_step)):
            return control

        rows = []
        examples = list(self.raw_eval_dataset)

        total_batches = (len(examples) + self.batch_size - 1) // self.batch_size

        for batch_examples in tqdm(
            batchify(examples, self.batch_size),
            desc="Evaluating",
            total=total_batches,
        ):
            prompts = [self.task_adapter.format_prompt(ex) for ex in batch_examples]
            generations = generate_batch(
                model=model,
                tokenizer=self.tokenizer,
                prompts=prompts,
                max_new_tokens=self.max_new_tokens,
            )

            for ex, gen_text in zip(batch_examples, generations):
                gold = self.task_adapter.extract_gold_answer(ex)
                pred = self.task_adapter.extract_predicted_answer(gen_text)

                rows.append(
                    {
                        "gold": gold,
                        "pred": pred,
                        "generated_text": gen_text,
                        "correct": pred == gold,
                    }
                )

        task_metrics = self.task_adapter.compute_rows_metrics(rows)
        log_dict = {f"eval_{key}": value for key, value in task_metrics.items()}

        if metrics is not None:
            metrics.update(log_dict)

        wandb_log_monotonic(log_dict, step=int(state.global_step))

        return control


class RoutingMetricsLoggingCallback(TrainerCallback):
    def __init__(self, routing_callback=None, moe_metrics_collector=None):
        self.routing_callback = routing_callback
        self.moe_metrics_collector = moe_metrics_collector

    def _current_k_metrics(self):
        if self.routing_callback is None:
            return {}
        current_k = getattr(self.routing_callback, "current_topk", None)
        if current_k is None:
            return {}
        return {"curriculum/current_k": int(current_k)}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return control

        custom_metrics = {}
        custom_metrics.update(self._current_k_metrics())
        if self.moe_metrics_collector is not None:
            custom_metrics.update(self.moe_metrics_collector.flush())

        logs.update(custom_metrics)

        wandb_log_monotonic(
            {
                key: value
                for key, value in custom_metrics.items()
                if isinstance(value, (int, float))
            },
            step=int(state.global_step),
        )
        return control


class ValidationTrackingCallback(TrainerCallback):
    def __init__(self, metric_name: str, output_dir: str, routing_callback=None):
        self.metric_name = str(metric_name)
        self.output_dir = Path(output_dir)
        self.routing_callback = routing_callback
        self.best_val_accuracy = None
        self.best_step = None
        self.eval_history = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return control

        metric_value = metrics.get(self.metric_name)
        step = int(state.global_step)
        current_k = None
        if self.routing_callback is not None:
            current_k = getattr(self.routing_callback, "current_topk", None)

        entry = {"step": step, "k": None if current_k is None else int(current_k)}
        if metric_value is not None:
            entry["val_accuracy"] = float(metric_value)
            if self.best_val_accuracy is None or metric_value > self.best_val_accuracy:
                self.best_val_accuracy = float(metric_value)
                self.best_step = step

        exact_eval_log = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                exact_eval_log[key] = value
        if current_k is not None:
            exact_eval_log["curriculum/current_k"] = int(current_k)
        append_exact_eval_log(state, exact_eval_log)

        self.eval_history.append(entry)
        return control

    def get_summary(
        self,
        test_accuracy_at_best,
        test_accuracy_at_final,
        train_runtime,
        final_step: int,
    ) -> dict:
        return {
            "best_val_accuracy": self.best_val_accuracy,
            "best_step": self.best_step,
            "test_accuracy_at_best": test_accuracy_at_best,
            "test_accuracy_at_final": test_accuracy_at_final,
            "train_runtime": None if train_runtime is None else float(train_runtime),
            "final_step": int(final_step),
            "curriculum_k_at_each_eval_step": self.eval_history,
        }

    def write_summary(self, summary: dict):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / "run_summary.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return out_path
