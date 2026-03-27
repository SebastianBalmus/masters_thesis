import re
import hashlib
import random

from datasets import ClassLabel, load_dataset

from data.base import BaseTaskAdapter

_LABEL_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)


class SciQTaskAdapter(BaseTaskAdapter):
    CHOICE_LABELS = ["A", "B", "C", "D"]

    # Fixed thresholds from EDA
    SUPPORT_SPLIT_1 = 182
    SUPPORT_SPLIT_2 = 419

    def load_raw(self):
        return load_dataset("allenai/sciq")

    def add_difficulty(self, ds):
        # Not used directly; split() handles annotation.
        return ds

    def _example_seed(self, example) -> int:
        key = f"{example['question']}|||{example['correct_answer']}|||{example['support']}"
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def _build_choices(self, example):
        choices = [
            {"text": example["correct_answer"].strip(), "is_correct": True},
            {"text": example["distractor1"].strip(), "is_correct": False},
            {"text": example["distractor2"].strip(), "is_correct": False},
            {"text": example["distractor3"].strip(), "is_correct": False},
        ]

        rng = random.Random(self._example_seed(example))
        rng.shuffle(choices)

        labeled_choices = []
        correct_label = None

        for label, choice in zip(self.CHOICE_LABELS, choices):
            labeled_choices.append((label, choice["text"]))
            if choice["is_correct"]:
                correct_label = label

        if correct_label is None:
            raise RuntimeError("Failed to assign a correct label for SciQ example.")

        return labeled_choices, correct_label

    def _estimate_difficulty_batch(self, batch):
        difficulty = []
        difficulty_name = []
        difficulty_score = []

        for support in batch["support"]:
            support_text = support.strip() if support is not None else ""
            support_len = len(support_text)
            difficulty_score.append(float(support_len))

            if support_len <= self.SUPPORT_SPLIT_1:
                difficulty.append(0)
                difficulty_name.append("easy")
            elif support_len <= self.SUPPORT_SPLIT_2:
                difficulty.append(1)
                difficulty_name.append("medium")
            else:
                difficulty.append(2)
                difficulty_name.append("hard")

        return {
            "difficulty_score": difficulty_score,
            "difficulty": difficulty,
            "difficulty_name": difficulty_name,
            "source_subset": ["SciQ"] * len(batch["support"]),
        }

    def split(self, ds):
        train_ds = ds["train"]
        val_ds = ds["validation"]

        train_ds = train_ds.map(self._estimate_difficulty_batch, batched=True)
        val_ds = val_ds.map(self._estimate_difficulty_batch, batched=True)

        train_ds = train_ds.cast_column(
            "difficulty",
            ClassLabel(names=["easy", "medium", "hard"]),
        )
        val_ds = val_ds.cast_column(
            "difficulty",
            ClassLabel(names=["easy", "medium", "hard"]),
        )

        return {
            "train": train_ds,
            "validation": val_ds,
        }

    def format_prompt(self, example):
        choices, _ = self._build_choices(example)
        choice_lines = [f"{label}. {text}" for label, text in choices]
        choices_block = "\n".join(choice_lines)

        return (
            f"Question: {example['question'].strip()}\n\n"
            f"Choices:\n{choices_block}\n\n"
            f"Answer:"
        )

    def format_completion(self, example):
        _, correct_label = self._build_choices(example)
        return f" {correct_label}{self.tokenizer.eos_token}"

    def has_task_metrics(self) -> bool:
        return True

    def get_metric_key(self) -> str:
        return "accuracy"

    def extract_gold_answer(self, example: dict) -> str:
        _, correct_label = self._build_choices(example)
        return correct_label

    def extract_predicted_answer(self, generated_text: str):
        text = generated_text.strip().upper()
        matches = _LABEL_RE.findall(text)
        if matches:
            return matches[0]
        if text and text[0] in {"A", "B", "C", "D"}:
            return text[0]
        return None

    def compute_rows_metrics(self, rows: list[dict]) -> dict:
        total = len(rows)
        correct = sum(int(row["correct"]) for row in rows)
        return {
            "accuracy": correct / total if total else 0.0,
        }
