from datasets import load_dataset
from data.base import BaseTaskAdapter


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
        for c in counts:
            if c <= 3:
                difficulty.append(0)
            elif c <= 5:
                difficulty.append(1)
            else:
                difficulty.append(2)

        return {"difficulty": difficulty}

    def add_difficulty(self, ds):
        ds = ds.map(self._estimate_difficulty_batch, batched=True)
        return ds

    def split(self, ds):
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
