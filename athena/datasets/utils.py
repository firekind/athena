from typing import Tuple, Callable

import numpy as np
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader


def train_val_split(
    batch_size: int,
    dataset: Dataset,
    val_split: float,
    shuffle: bool = True,
    num_workers: int = 4,
    collate_fn: Callable = None,
    pin_memory: bool = True,
    drop_last: bool = False,
    timeout: float = 0,
    worker_init_fn: Callable = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Splits the dataset into train and validation dataloaders.

    Args:
        batch_size (int): The batch size
        dataset (Dataset): The dataset to split
        val_split (float): The amount of data from the original dataset to be used \
            in the validation dataloader. Should be a value between 0 and 1.
        shuffle (bool, optional): If ``True``, the train and validation data are chosen at \
            random. Defaults to ``True``.
        num_workers (int, optional): How many subprocesses to use for data loading. Defaults to 4.
        collate_fn (Callable, optional): Merges a list of samples to form a mini-batch of Tensor(s). \
            Used when using batched loading from a map-style dataset. Defaults to ``None``.
        pin_memory (bool, optional): If ``True``, copies Tensors to cuda pinned memory. Defaults to ``True``.
        drop_last (bool, optional): Set to ``True`` to drop the last incomplete batch. Defaults to ``False``.
        timeout (float, optional): If positive, the timeout value for collecting a batch from workers. \
            Should always be non-negative. Defaults to 0.
        worker_init_fn (Callable, optional): If not None, this will be called on each worker subprocess with \
            the worker id (an int in [0, num_workers - 1]) as input, after seeding and before data loading.

    Returns:
        Tuple[DataLoader, DataLoader]: The train dataloader and the validation dataloader
    """
    # getting dataset size
    dataset_size = len(dataset)

    # generating indices
    indices = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(indices)

    # getting amount of data in validation split
    split = np.floor(val_split * dataset_size).astype(np.int)

    # generating samplers
    train_sampler = SubsetRandomSampler(indices[split:])
    val_sampler = SubsetRandomSampler(indices[:split])

    # creating and returning dataloaders
    return (
        DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        ),
        DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        ),
    )
