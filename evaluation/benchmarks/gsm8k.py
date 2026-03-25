import re
from datasets import load_dataset

from evaluation.benchmarks.base import BaseBenchmarkAdapter


_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def normalize_number_string(text: str) -> str:
    text = text.strip()
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("%", "")
    text = text.replace("−", "-")
    return text


class GSM8KBenchmarkAdapter(BaseBenchmarkAdapter):
    def load_split(self):
        split = self.cfg.eval.split
        return load_dataset("openai/gsm8k", "main", split=split)

    def format_prompt(self, example: dict) -> str:
        # Zero-shot prompt aligned with your training/eval style
        return f"Q: {example['question'].strip()}\nA:"

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

    def compute_metrics(self, rows: list[dict]) -> dict:
        total = len(rows)
        correct = sum(int(row["correct"]) for row in rows)
        accuracy = correct / total if total else 0.0

        return {
            "num_examples": total,
            "num_correct": correct,
            "accuracy_extracted_answer": accuracy,
        }
