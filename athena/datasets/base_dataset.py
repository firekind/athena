from abc import ABC, abstractmethod
from typing import Iterable, Callable, Union

from torch.utils.data import Sampler, DataLoader


class BaseDataset(ABC):
    def __init__(self):
        """
        The base class for datasets. Enables the use of a builder style API.

        Parameters that can be set using builder API:
            root (str, optional): Root directory of dataset. Defaults to "./data".
            train (bool, optional): If True, creates training dataset. Defaults to True.
            transform (transforms.Compose, optional): Transform to apply to training data. Defaults to None.
            target_transform (transforms.Compose, optional): Transform to apply to the targets of the training data. Defaults to None.
            download (bool, optional): If True, downloads the dataset. Defaults to True.
            batch_size (int, optional): The batch size. Defaults to 64.
            shuffle (bool, optional): If True, shuffles the dataset. Defaults to True.
            sampler (Union[Sampler, Iterable], optional): Defines the strategy to draw samples from the dataset. If specified, shuffle must not be specified. Defaults to None.
            batch_sampler (Union[Sampler, Iterable], optional): Like sampler, but returns a batch of indices at a time. Mutually exclusive with batch_size, shuffle, sampler, and drop_last. Defaults to None.
            num_workers (int, optional): How many subprocesses to use for data loading. Defaults to 4.
            collate_fn (Callable, optional): Merges a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a map-style dataset. Defaults to None.
            pin_memory (bool, optional): If True, copies Tensors to cuda pinned memory. Defaults to True.
            drop_last (bool, optional): Set to True to drop the last incomplete batch. Defaults to False.
            timeout (float, optional): If positive, the timeout value for collecting a batch from workers. Should always be non-negative. Defaults to 0.
            worker_init_fn (Callable, optional): If not None, this will be called on each worker subprocess with the worker id (an int in [0, num_workers - 1]) as input, after seeding and before data loading. Defaults to None.
            use_default_transforms (bool, optional): If true, will use the default train and test transforms specified in this module.
        """
        self._root = "./data"
        self._train = True
        self._transform = None
        self._target_transform = None
        self._download = True
        self._batch_size = 64
        self._shuffle = True
        self._sampler = None
        self._batch_sampler = None
        self._num_workers = 4
        self._collate_fn = None
        self._pin_memory = True
        self._drop_last = False
        self._timeout = 0
        self._worker_init_fn = None
        self._use_default_transforms = False

    def root(self, path: str):
        self._root = root
        return self

    def train(self, train: bool = True):
        self._train = train
        return self

    def test(self):
        self._train = False
        return self

    def transform(self, transform: Callable):
        self._transform = transform
        return self

    def target_transform(self, target_transform: Callable):
        self._target_transform = target_transform
        return self

    def download(self, download: bool):
        self._download = download
        return self

    def batch_size(self, batch_size: int):
        self._batch_size = batch_size
        return self

    def shuffle(self, shuffle: bool):
        self._shuffle = shuffle
        return self

    def sampler(self, sampler: Union[Sampler, Iterable]):
        self._sampler = sampler
        return self

    def batch_sampler(self, batch_sampler: Union[Sampler, Iterable]):
        self._batch_sampler = batch_sampler
        return self

    def num_workers(self, num_workers: int):
        self._num_workers = num_workers
        return self

    def collate_fn(self, collate_fn: Callable):
        self._collate_fn = collate_fn
        return self

    def pin_memory(self, pin_memory: bool):
        self._pin_memory = pin_memory
        return self

    def drop_last(self, drop_last: bool):
        self._drop_last = drop_last
        return self

    def timeout(self, timeout: float):
        self._timeout = timeout
        return self

    def worker_init_fn(self, worker_init_fn: Callable):
        self._worker_init_fn = worker_init_fn
        return self

    def use_default_transforms(self, use_default_transforms: bool = True):
        self._use_default_transforms = use_default_transforms
        return self

    @abstractmethod
    def build(self) -> DataLoader:
        """
        Builds the dataset and returns the dataloader.
        """

        assert not (
            self._use_default_transforms and self._transform
        ), "Specify only `use_default_transforms` or `transform`, not both."

        if self._use_default_transforms:
            if self._train:
                self._transform = self._default_train_transform()
            else:
                self._transform = self._default_test_transform()

    @abstractmethod
    def _default_train_transform(self) -> Callable:
        """
        The default train transforms.

        Returns:
            Callable: The transform.
        """

    @abstractmethod
    def _default_test_transform(self) -> Callable:
        """
        The default test transforms.

        Returns:
            Callable: The transforms.
        """
