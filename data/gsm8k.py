import re

from datasets import ClassLabel, load_dataset
from data.base import BaseTaskAdapter


_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def normalize_number_string(text: str) -> str:
    text = text.strip()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("%", "")
    text = text.replace("−", "-")
    return text


class GSM8KTaskAdapter(BaseTaskAdapter):
    def __init__(self, cfg, tokenizer):
        super().__init__(cfg, tokenizer)

    def load_raw(self):
        return load_dataset(self.cfg.dataset_id, "main", split="train")

    def _estimate_difficulty_batch(self, batch: dict) -> dict:
        counts = []
        for text in batch["answer"]:
            rationale = text.split("####")[0].strip()
            counts.append(len(rationale.split("\n")))

        difficulty = []
        difficulty_name = []
        for c in counts:
            if c <= 3:
                difficulty.append(0)
                difficulty_name.append("easy")
            elif c <= 5:
                difficulty.append(1)
                difficulty_name.append("medium")
            else:
                difficulty.append(2)
                difficulty_name.append("hard")

        return {
            "difficulty": difficulty,
            "difficulty_name": difficulty_name,
            "source_subset": ["GSM8K"] * len(batch["answer"]),
        }

    def add_difficulty(self, ds):
        ds = ds.map(self._estimate_difficulty_batch, batched=True)
        ds = ds.cast_column(
            "difficulty",
            ClassLabel(names=["easy", "medium", "hard"]),
        )
        return ds

    def split(self, ds):
        ds = self.add_difficulty(ds)

        split_ds = ds.train_test_split(
            test_size=self.cfg.data.test_size if "data" in self.cfg else 0.1,
            seed=self.cfg.seed,
            stratify_by_column="difficulty",
        )
        return {
            "train": split_ds["train"],
            "validation": split_ds["test"],
        }

    def format_prompt(self, example):
        return f"Q: {example['question'].strip()}\nA:"

    def format_completion(self, example):
        return example["answer"].strip() + self.tokenizer.eos_token

    def has_task_metrics(self) -> bool:
        return True

    def get_metric_key(self) -> str:
        return "accuracy_extracted_answer"

    def extract_gold_answer(self, example: dict) -> str:
        answer_text = example["answer"]
        if "####" in answer_text:
            return normalize_number_string(answer_text.split("####")[-1].strip())
        return normalize_number_string(answer_text.strip())

    def extract_predicted_answer(self, generated_text: str):
        matches = _NUMBER_RE.findall(generated_text)
        if not matches:
            return None
        return normalize_number_string(matches[-1])

    def compute_rows_metrics(self, rows: list[dict]) -> dict:
        total = len(rows)
        correct = sum(int(row["correct"]) for row in rows)

        return {
            "accuracy_extracted_answer": correct / total if total else 0.0,
        }
