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
