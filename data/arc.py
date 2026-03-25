from datasets import ClassLabel, concatenate_datasets, load_dataset

from data.base import BaseTaskAdapter


NEGATION_WORDS = {"not", "except", "least", "never", "incorrect"}
NUMERIC_CUES = {"percent", "percentage", "sum", "difference", "total", "average"}
CAUSAL_CUES = {"why", "cause", "best explains", "most likely", "reason"}


class ARCTaskAdapter(BaseTaskAdapter):
    def load_raw(self):
        easy = load_dataset(self.cfg.dataset_id, self.cfg.dataset_easy_subset)
        challenge = load_dataset(self.cfg.dataset_id, self.cfg.dataset_challenge_subset)
        return {"easy": easy, "challenge": challenge}

    def add_difficulty(self, ds):
        # Not used directly; split() handles per-source annotation.
        return ds

    def _challenge_difficulty_batch(self, batch):
        levels = []

        for question, choices in zip(batch["question"], batch["choices"]):
            q = question.strip().lower()
            q_len = len(q.split())

            option_texts = choices["text"]
            avg_choice_len = sum(len(c.split()) for c in option_texts) / max(
                1, len(option_texts)
            )

            score = 0.0
            score += 0.45 * q_len
            score += 0.35 * avg_choice_len

            if any(word in q for word in NEGATION_WORDS):
                score += 4.0

            if any(word in q for word in NUMERIC_CUES):
                score += 2.5

            if any(phrase in q for phrase in CAUSAL_CUES):
                score += 2.0

            levels.append(score)

        return {"difficulty_score": levels}

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
        return f" {example['answerKey']}{self.tokenizer.eos_token}"

    def split(self, ds):
        easy_train = ds["easy"]["train"]
        challenge_train = ds["challenge"]["train"]
        challenge_val = ds["challenge"]["validation"]

        # Stage 0: all ARC-Easy
        easy_train = easy_train.map(
            lambda ex: {
                "difficulty": 0,
                "difficulty_name": "arc_easy",
                "source_subset": "ARC-Easy",
            }
        )

        # Score challenge train examples
        challenge_train = challenge_train.map(
            self._challenge_difficulty_batch,
            batched=True,
        )

        scores = challenge_train["difficulty_score"]
        sorted_scores = sorted(scores)
        median_score = sorted_scores[len(sorted_scores) // 2]

        def assign_challenge_bucket(example):
            level = 1 if example["difficulty_score"] <= median_score else 2
            name = "challenge_easy" if level == 1 else "challenge_hard"
            return {
                "difficulty": level,
                "difficulty_name": name,
                "source_subset": "ARC-Challenge",
            }

        challenge_train = challenge_train.map(assign_challenge_bucket)

        # Validation uses challenge only; mark but do not care much about curriculum there.
        challenge_val = challenge_val.map(
            self._challenge_difficulty_batch,
            batched=True,
        )
        challenge_val = challenge_val.map(
            lambda ex: {
                "difficulty": 1 if ex["difficulty_score"] <= median_score else 2,
                "difficulty_name": "challenge_val",
                "source_subset": "ARC-Challenge",
            }
        )

        train_ds = concatenate_datasets([easy_train, challenge_train])
        train_ds = train_ds.cast_column(
            "difficulty",
            ClassLabel(names=["arc_easy", "challenge_easy", "challenge_hard"]),
        )

        challenge_val = challenge_val.cast_column(
            "difficulty",
            ClassLabel(names=["arc_easy", "challenge_easy", "challenge_hard"]),
        )

        return {
            "train": train_ds,
            "validation": challenge_val,
        }
