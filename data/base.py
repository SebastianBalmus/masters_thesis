from abc import ABC, abstractmethod
from datasets import Dataset, DatasetDict


class BaseTaskAdapter(ABC):
    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer

    @abstractmethod
    def load_raw(self) -> Dataset:
        pass

    @abstractmethod
    def split(self, ds: Dataset) -> DatasetDict:
        pass

    @abstractmethod
    def format_prompt(self, example: dict) -> str:
        pass

    @abstractmethod
    def format_completion(self, example: dict) -> str:
        pass

    def has_task_metrics(self) -> bool:
        return False

    def get_metric_key(self) -> str:
        return "eval_loss"

    def compute_generative_metrics(
        self, model, tokenizer, dataset, batch_size: int
    ) -> dict:
        raise NotImplementedError("This task does not implement task-specific metrics.")

    def tokenize_example(self, example: dict) -> dict:
        prompt_text = self.format_prompt(example)
        completion_text = self.format_completion(example)

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        completion_ids = self.tokenizer(
            completion_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        attention_mask = [1] * len(input_ids)

        max_seq_length = self.cfg.max_seq_length
        input_ids = input_ids[:max_seq_length]
        attention_mask = attention_mask[:max_seq_length]
        labels = labels[:max_seq_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def build_splits(self):
        raw = self.load_raw()
        split_ds = self.split(raw)

        tokenized_train = split_ds["train"].map(
            self.tokenize_example,
            remove_columns=split_ds["train"].column_names,
        )
        tokenized_val = split_ds["validation"].map(
            self.tokenize_example,
            remove_columns=split_ds["validation"].column_names,
        )

        return {
            "train_raw": split_ds["train"],
            "validation_raw": split_ds["validation"],
            "train_tokenized": tokenized_train,
            "validation_tokenized": tokenized_val,
        }
