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
    def add_difficulty(self, ds: Dataset) -> Dataset:
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
            "difficulty": example["difficulty"],
        }

    def build_tokenized_splits(self):
        ds = self.load_raw()
        ds = self.add_difficulty(ds)
        split_ds = self.split(ds)

        train_ds = split_ds["train"].map(
            self.tokenize_example,
            remove_columns=split_ds["train"].column_names,
        )
        val_ds = split_ds["validation"].map(
            self.tokenize_example,
            remove_columns=split_ds["validation"].column_names,
        )
        return train_ds, val_ds
