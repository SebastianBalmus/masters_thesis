import hashlib
import random
import re

from datasets import load_dataset

from evaluation.benchmarks.base import BaseBenchmarkAdapter


_LABEL_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)


class SciQBenchmarkAdapter(BaseBenchmarkAdapter):
    CHOICE_LABELS = ["A", "B", "C", "D"]

    def load_split(self):
        return load_dataset("allenai/sciq", split=self.cfg.eval.split)

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

    def format_prompt(self, example: dict) -> str:
        choices, _ = self._build_choices(example)
        choice_lines = [f"{label}. {text}" for label, text in choices]
        choices_block = "\n".join(choice_lines)

        return (
            f"Question: {example['question'].strip()}\n\n"
            f"Choices:\n{choices_block}\n\n"
            f"Answer:"
        )

    def extract_gold_answer(self, example: dict) -> str:
        _, correct_label = self._build_choices(example)
        return correct_label

    def extract_predicted_answer(self, generated_text: str):
        text = generated_text.strip().upper()

        matches = _LABEL_RE.findall(text)
        if matches:
            return matches[0]

        if text:
            first = text[0]
            if first in {"A", "B", "C", "D"}:
                return first

        return None

    def compute_metrics(self, rows: list[dict]) -> dict:
        total = len(rows)
        correct = sum(int(row["correct"]) for row in rows)
        accuracy = correct / total if total else 0.0

        return {
            "num_examples": total,
            "num_correct": correct,
            "accuracy": accuracy,
        }
