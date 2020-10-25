from copy import deepcopy
from typing import Callable, Tuple, Union

import numpy as np
import torch
from athena.utils.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler


def train_val_split(
    dataset: Dataset,
    batch_size: int,
    val_split: float,
    shuffle: bool = True,
    num_workers: int = 4,
    collate_fn: Callable = None,
    pin_memory: bool = True,
    drop_last: bool = False,
    timeout: float = 0,
    worker_init_fn: Callable = None,
    val_transform: Callable = None,
    target_val_transform: Callable = None,
    use_default_val_transform: bool = False,
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
        val_transform (Callable, optional): The transform to be used in the val dataset. Defaults to None.
        target_val_transform (Callable, optional): The transform to be used for the targets in the val dataset. \
            Defaults to None.
        use_default_val_transform (bool, optional): Whether to use the default val transforms or not. \
            Defaults to False.

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

    # creating val dataset
    val_dataset = deepcopy(dataset)

    # setting the transform for the val dataset
    if use_default_val_transform:
        val_transform = val_dataset.default_val_transform()

    val_dataset.train = False
    val_dataset.transform = val_transform
    val_dataset.target_transform = target_val_transform

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
            val_dataset,
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


def calculate_mean_and_std(
    data_gen: Union[DataLoader, Dataset], device: str = "best"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the mean and standard deviation of a image based dataset (or dataloader)

    Args:
        data_gen (Union[DataLoader, Dataset]): The dataset or dataloader.
        device (str, optional): A valid pytorch device string. If "best", automatically chooses the best device. \
            Defaults to "best".

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The mean and standard deviation.
    """
    mean = 0.0
    std = 0.0
    num_samples = 0.0

    if device == "best":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    for data, target in data_gen:
        data = ToTensor(data).to(device)

        if isinstance(data_gen, DataLoader):
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            num_samples += batch_samples
        else:
            data = data.view(data.size(0), -1)
            mean += data.mean(1)
            std += data.std(1)
            num_samples += 1

    return mean / num_samples, std / num_samples
