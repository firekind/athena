from typing import Iterable, Callable, Union

import torch
from torchvision import datasets
from torchvision import transforms


def MNIST(
    root: str = "./data",
    train: bool = True,
    transform: transforms.Compose = None,
    target_transform: transforms.Compose = None,
    download: bool = False,
    batch_size: int = 64,
    shuffle: bool = True,
    sampler: Union[torch.utils.data.Sampler, Iterable] = None,
    batch_sampler: Union[torch.utils.data.Sampler, Iterable] = None,
    num_workers: int = 4,
    collate_fn: Callable = None,
    pin_memory: bool = True,
    drop_last: bool = False,
    timeout: float = 0,
    worker_init_fn: Callable = None,
) -> torch.utils.data.DataLoader:
    """
    Creates an MNIST dataloder given the arguments.

    Args:
        root (str, optional): Root directory of dataset. Defaults to "./data".
        train (bool, optional): If True, creates training dataset. Defaults to True.
        transform (transforms.Compose, optional): Transform to apply to training data. Defaults to None.
        target_transform (transforms.Compose, optional): Transform to apply to the targets of the training data. Defaults to None.
        download (bool, optional): If True, downloads the dataset. Defaults to False.
        batch_size (int, optional): The batch size. Defaults to 64.
        shuffle (bool, optional): If True, shuffles the dataset. Defaults to True.
        sampler (Union[torch.utils.data.Sampler, Iterable], optional): Defines the strategy to draw samples from the dataset. If specified, shuffle must not be specified. Defaults to None.
        batch_sampler (Union[torch.utils.data.Sampler, Iterable], optional): Like sampler, but returns a batch of indices at a time. Mutually exclusive with batch_size, shuffle, sampler, and drop_last. Defaults to None.
        num_workers (int, optional): How many subprocesses to use for data loading. Defaults to 4.
        collate_fn (Callable, optional): Merges a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a map-style dataset. Defaults to None.
        pin_memory (bool, optional): If True, copies Tensors to cuda pinned memory. Defaults to True.
        drop_last (bool, optional): Set to True to drop the last incomplete batch. Defaults to False.
        timeout (float, optional): If positive, the timeout value for collecting a batch from workers. Should always be non-negative. Defaults to 0.
        worker_init_fn (Callable, optional): If not None, this will be called on each worker subprocess with the worker id (an int in [0, num_workers - 1]) as input, after seeding and before data loading. Defaults to None.

    Returns:
        torch.utils.data.DataLoader: [description]
    """

    dataset = datasets.MNIST(
        root=root,
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn
    )
