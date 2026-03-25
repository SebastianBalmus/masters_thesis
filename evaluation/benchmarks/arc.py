import re
from datasets import load_dataset

from evaluation.benchmarks.base import BaseBenchmarkAdapter


_LABEL_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)


class ARCBenchmarkAdapter(BaseBenchmarkAdapter):
    def load_split(self):
        # Uses the official ARC-Challenge split from HF
        return load_dataset(
            "allenai/ai2_arc", "ARC-Challenge", split=self.cfg.eval.split
        )

    def format_prompt(self, example: dict) -> str:
        labels = example["choices"]["label"]
        texts = example["choices"]["text"]

        choice_lines = [f"{label}. {text}" for label, text in zip(labels, texts)]
        choices_block = "\n".join(choice_lines)

        return (
            f"Question: {example['question'].strip()}\n\n"
            f"Choices:\n{choices_block}\n\n"
            f"Answer:"
        )

    def extract_gold_answer(self, example: dict) -> str:
        return str(example["answerKey"]).strip().upper()

    def extract_predicted_answer(self, generated_text: str):
        text = generated_text.strip().upper()

        # First try exact single-letter answer anywhere in the output.
        matches = _LABEL_RE.findall(text)
        if matches:
            return matches[0]

        # Fallback: accept things like "A.", "(B)", "C:" at the start.
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
