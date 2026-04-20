import re

from datasets import concatenate_datasets, load_dataset

from data.base import BaseTaskAdapter

_LABEL_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)


class ARCTaskAdapter(BaseTaskAdapter):
    def load_raw(self):
        easy = load_dataset(self.cfg.dataset_id, self.cfg.dataset_easy_subset)
        challenge = load_dataset(self.cfg.dataset_id, self.cfg.dataset_challenge_subset)
        return {"easy": easy, "challenge": challenge}

    def _format_arc_row(self, example):
        labels = example["choices"]["label"]
        texts = example["choices"]["text"]
        choice_lines = [f"{lbl}. {txt}" for lbl, txt in zip(labels, texts)]
        choices_block = "\n".join(choice_lines)

        return (
            f"Question: {example['question'].strip()}\n\n"
            f"Choices:\n{choices_block}\n\n"
            f"Answer:"
        )

    def format_prompt(self, example):
        return self._format_arc_row(example)

    def format_completion(self, example):
        return f" {str(example['answerKey']).strip().upper()}{self.tokenizer.eos_token}"

    def split(self, ds):
        easy_train = ds["easy"]["train"]
        challenge_train = ds["challenge"]["train"]
        challenge_val = ds["challenge"]["validation"]

        train_ds = concatenate_datasets([easy_train, challenge_train])

        return {
            "train": train_ds,
            "validation": challenge_val,
        }

    # ---- Task metric API ----

    def has_task_metrics(self) -> bool:
        return True

    def get_metric_key(self) -> str:
        return "accuracy"

    def extract_gold_answer(self, example: dict) -> str:
        return str(example["answerKey"]).strip().upper()

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

    def compute_rows_metrics(self, rows: list[dict]) -> dict:
        total = len(rows)
        correct = sum(int(row["correct"]) for row in rows)

        return {
            "accuracy": correct / total if total else 0.0,
        }
