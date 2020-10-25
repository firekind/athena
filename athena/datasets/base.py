from abc import abstractmethod
from typing import Callable, Iterable, Union, Type

from torch.utils.data import DataLoader, Sampler, Dataset


class BaseDataset(Dataset):
    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        transform: Callable = None,
        target_transform: Callable = None,
        download: bool = False,
        use_default_transforms: bool = False,
        **kwargs
    ):
        """
        Base class for datasets.

        Args:
            root (str, optional): The root directory of the dataset. Defaults to "./data".
            train (bool, optional): Whether its train or test dataset. Defaults to ``True``.
            transform (Callable, optional): The tranform to apply on the data. Defaults to ``None``.
            target_transform (Callable, optional): The transform to apply on the labels. Defaults \
                to ``None``.
            download (bool, optional): Whether the dataset should be downloaded or not. Defaults \
                to ``False``.
            use_default_transforms (bool, optional): Whether the default transforms must be used \
                or not. Defaults to ``False``.
        """

        super(BaseDataset, self).__init__()

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.use_default_transforms = use_default_transforms

        self._set_transforms()

        for name, value in kwargs.items():
            setattr(self, name, value)

    def _set_transforms(self) -> DataLoader:
        """
        Sets the transform for the dataset, if ``use_default_transforms`` is enabled.
        """

        assert not (
            self.use_default_transforms and self.transform
        ), "Specify only `use_default_transforms` or `transform`, not both."

        if self.use_default_transforms:
            if self.train:
                self.transform = self.default_train_transform()
            else:
                self.transform = self.default_val_transform()

    def default_train_transform(self):
        pass

    def default_val_transform(self):
        pass

    @classmethod
    def builder(cls):
        return DataLoaderBuilder(cls)


class DataLoaderBuilder:
    def __init__(self, dataset_cls: Type[BaseDataset]):
        """
        Class used to build a dataloader using the builder pattern.

        Default values of various parameters (if not set using the builder API):
            * **batch_size** *(int)*: 64.

            * **shuffle** *(bool)*: ``True``.

            * **sampler** *(Union[Sampler, Iterable])*: ``None``.

            * **batch_sampler** *(Union[Sampler, Iterable])*: ``None``.

            * **num_workers** *(int)*: 4.

            * **collate_fn** *(Callable)*: ``None``.

            * **pin_memory** *(bool)*: ``True``.

            * **drop_last** *(bool)*: ``False``.

            * **timeout** *(float)*: 0.

            * **worker_init_fn** *(Callable)*: ``None``.
        """

        self.dataset_cls = dataset_cls
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
        self.dataset_args = {}

    def build(self) -> DataLoader:
        """
        Builds the dataset and returns a pytorch ``DataLoader``.

        Returns:
            DataLoader: The cifar10 ``DataLoader``.
        """

        return DataLoader(
            self.dataset_cls(**self.dataset_args),
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            sampler=self._sampler,
            batch_sampler=self._batch_sampler,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
            drop_last=self._drop_last,
            timeout=self._timeout,
            worker_init_fn=self._worker_init_fn,
        )

    def root(self, path: str) -> "BaseDataset":
        """
        Sets the ``root``. Used in the builder API

        Args:
            path (str): Root directory of dataset.

        Returns:
            BaseDataset: Object of this class.
        """
        self.dataset_args["root"] = path
        return self

    def train(self, train: bool = True) -> "BaseDataset":
        """
        Sets the train data flag. Used in the builder API

        Args:
            train (str, optional): If ``True``, creates training dataset. Defaults to ``True``.

        Returns:
            BaseDataset: Object of this class.
        """
        self.dataset_args["train"] = train
        return self

    def test(self) -> "BaseDataset":
        """
        Sets the train data flag to ``False``, so that a test dataset is created. Used in the \
            builder API.

        Returns:
            BaseDataset: Object of this class.
        """
        self.dataset_args["train"] = False
        return self

    def transform(self, transform: Callable) -> "BaseDataset":
        """
        Sets the transforms to use. Used in the builder API

        Args:
            transform (Callable): Transform to apply to training data.

        Returns:
            BaseDataset: Object of this class.
        """
        self.dataset_args["transform"] = transform
        return self

    def target_transform(self, target_transform: Callable) -> "BaseDataset":
        """
        Sets the transforms to use on the target data (labels). Used in the builder API

        Args:
            target_transform (Callable): Transform to apply to training data's targets.

        Returns:
            BaseDataset: Object of this class.
        """
        self.dataset_args["target_transform"] = target_transform
        return self

    def download(self, download: bool) -> "BaseDataset":
        """
        Sets the ``download`` flag. Used in the builder API

        Args:
            download (bool): If True, downloads the dataset.

        Returns:
            BaseDataset: Object of this class.
        """
        self.dataset_args["download"] = download
        return self

    def batch_size(self, batch_size: int) -> "BaseDataset":
        """
        Sets the batch size. Used in the builder API

        Args:
            batch_size (int): The batch size.

        Returns:
            BaseDataset: Object of this class.
        """
        self._batch_size = batch_size
        return self

    def shuffle(self, shuffle: bool) -> "BaseDataset":
        """
        Sets whether to shuffle the dataset. Used in the builder API

        Args:
            shuffle (bool): If True, shuffles the dataset.

        Returns:
            BaseDataset: Object of this class.
        """
        self._shuffle = shuffle
        return self

    def sampler(self, sampler: Union[Sampler, Iterable]) -> "BaseDataset":
        """
        Sets the ``sampler``. Used in the builder API

        Args:
            sampler (Union[Sampler, Iterable]): Defines the strategy to draw samples from the \
                dataset. If specified, shuffle must not be specified.

        Returns:
            BaseDataset: Object of this class.
        """
        self._sampler = sampler
        return self

    def batch_sampler(self, batch_sampler: Union[Sampler, Iterable]) -> "BaseDataset":
        """
        Sets the ``batch_sampler``. Used in the builder API

        Args:
            batch_sampler (Union[Sampler, Iterable]): Like sampler, but returns a batch of indices \
                at a time. Mutually exclusive with batch_size, shuffle, sampler, and drop_last.

        Returns:
            BaseDataset: Object of this class.
        """
        self._batch_sampler = batch_sampler
        return self

    def num_workers(self, num_workers: int) -> "BaseDataset":
        """
        Sets the number of workers to use. Used in the builder API

        Args:
            num_workers (int): How many subprocesses to use for data loading.

        Returns:
            BaseDataset: Object of this class.
        """
        self._num_workers = num_workers
        return self

    def collate_fn(self, collate_fn: Callable) -> "BaseDataset":
        """
        Sets the collate_fn. Used in the builder API

        Args:
            collate_fn (Callable): Merges a list of samples to form a mini-batch of Tensor(s). \
                Used when using batched loading from a map-style dataset.

        Returns:
            BaseDataset: Object of this class.
        """
        self._collate_fn = collate_fn
        return self

    def pin_memory(self, pin_memory: bool) -> "BaseDataset":
        """
        Sets whether to pin memory. Used in the builder API

        Args:
            pin_memory (bool): If True, copies Tensors to cuda pinned memory.

        Returns:
            BaseDataset: Object of this class.
        """
        self._pin_memory = pin_memory
        return self

    def drop_last(self, drop_last: bool) -> "BaseDataset":
        """
        Sets ``drop_last``. Used in the builder API

        Args:
            drop_last (bool): Set to True to drop the last incomplete batch.

        Returns:
            BaseDataset: Object of this class.
        """
        self._drop_last = drop_last
        return self

    def timeout(self, timeout: float) -> "BaseDataset":
        """
        Sets the ``timeout``. Used in the builder API

        Args:
            timeout (float): If positive, the timeout value for collecting a batch from workers. \
                Should always be non-negative.

        Returns:
            BaseDataset: Object of this class.
        """
        self._timeout = timeout
        return self

    def worker_init_fn(self, worker_init_fn: Callable) -> "BaseDataset":
        """
        Sets the ``worker_init_fn``. Used in the builder API

        Args:
            worker_init_fn (Callable): If not None, this will be called on each worker subprocess with \
                the worker id (an int in [0, num_workers - 1]) as input, after seeding and before data loading.

        Returns:
            BaseDataset: Object of this class.
        """
        self._worker_init_fn = worker_init_fn
        return self

    def use_default_transforms(
        self, use_default_transforms: bool = True
    ) -> "BaseDataset":
        """
        Sets the ``use_default_transforms``. Used in the builder API

        Args:
            batch_size (bool, optional): If true, will use the default train and test transforms specified \
                in :meth:`default_train_transform` and :meth:`default_val_transform`. Defaults to True.

        Returns:
            BaseDataset: Object of this class.
        """
        self.dataset_args["use_default_transforms"] = use_default_transforms
        return self
