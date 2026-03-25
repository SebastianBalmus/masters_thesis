import numpy as np
import torch
from torch.utils.data import IterableDataset


class RandomIterableDataset(IterableDataset):
    def __init__(self, dataset, seed=42):
        self.dataset = dataset
        self.seed = seed
        self.num_examples = len(dataset)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        rng = np.random.default_rng(self.seed + worker_id)

        while True:
            idx = int(rng.integers(0, self.num_examples))
            yield self.dataset[idx]


class CurriculumIterableDataset(IterableDataset):
    def __init__(self, dataset, curriculum_state, seed=42):
        self.dataset = dataset
        self.curriculum_state = curriculum_state
        self.seed = seed

        difficulties = dataset["difficulty"]
        self.indices_by_level = {
            level: [i for i, d in enumerate(difficulties) if d == level]
            for level in range(curriculum_state.num_difficulty_levels)
        }

        self.level_counts = {
            level: len(indices) for level, indices in self.indices_by_level.items()
        }

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        rng = np.random.default_rng(self.seed + worker_id)

        while True:
            level = self.curriculum_state.current_stage
            indices = self.indices_by_level[level]
            if not indices:
                raise RuntimeError(f"Curriculum level {level} is empty.")
            idx = int(rng.choice(indices))
            yield self.dataset[idx]
