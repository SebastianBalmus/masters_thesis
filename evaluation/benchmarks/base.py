from abc import ABC, abstractmethod


class BaseBenchmarkAdapter(ABC):
    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer

    @abstractmethod
    def load_split(self):
        pass

    @abstractmethod
    def format_prompt(self, example: dict) -> str:
        pass

    @abstractmethod
    def extract_gold_answer(self, example: dict) -> str:
        pass

    @abstractmethod
    def extract_predicted_answer(self, generated_text: str):
        pass

    @abstractmethod
    def compute_metrics(self, rows: list[dict]) -> dict:
        pass
