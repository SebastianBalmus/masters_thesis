import torch
from tqdm import tqdm
from transformers import TrainerCallback


def batchify(items, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def pin_and_move_to_device(
    batch_tensors: dict[str, torch.Tensor], device: torch.device
):
    moved = {}
    for key, value in batch_tensors.items():
        if value.device.type == "cpu":
            value = value.pin_memory()
        moved[key] = value.to(device, non_blocking=True)
    return moved


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

    try:
        outputs = model.generate(cache_implementation="static", **gen_kwargs)
    except Exception:
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
    ):
        self.task_adapter = task_adapter
        self.tokenizer = tokenizer
        self.raw_eval_dataset = raw_eval_dataset
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if model is None:
            return control

        if not self.task_adapter.has_task_metrics():
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

        try:
            import wandb

            if wandb.run is not None:
                wandb.log(log_dict, step=state.global_step)
        except Exception as e:
            print(f"Warning: failed to log eval metrics to wandb: {e}")

        return control
