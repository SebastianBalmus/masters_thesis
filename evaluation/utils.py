import math


def batchify(items, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def num_batches(n_items: int, batch_size: int) -> int:
    return math.ceil(n_items / batch_size)
