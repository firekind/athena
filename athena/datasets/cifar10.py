from typing import Callable, Tuple

import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .base_dataset import BaseDataset


class cifar10(BaseDataset):

    mean = (0.4914, 0.4822, 0.4465) #: mean of the dataset.
    std = (0.2023, 0.1994, 0.2010) #: std of the dataset.

    def __init__(self):
        """
        The cifar 10 dataset.
        """
        super(cifar10, self).__init__()

    def build(self) -> DataLoader:
        """
        Builds the dataset and returns a pytorch ``DataLoader``.

        Returns:
            DataLoader: The cifar10 ``DataLoader``.
        """
        super(cifar10, self).build()

        return DataLoader(
            _cifar10_dataset(
                root=self._root,
                train=self._train,
                transform=self._transform,
                target_transform=self._target_transform,
                download=self._download,
            ),
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

    def default_train_transform(self):
        """
        Default cifar10 training transforms. Performs

        * Normalization using mean: (0.4914, 0.4822, 0.4465), std: (0.2023, 0.1994, 0.2010)
        
        Returns:
            Callable: The transform, an ``albumentations.Compose`` object.
        """
        return A.Compose(
            [
                A.Normalize(
                    mean=cifar10.mean, std=cifar10.std
                ),  # Normalizing
            ]
        )

    def default_test_transform(self):
        """
        Default cifar10 test transforms. Performs

        * Normalization using mean: (0.4914, 0.4822, 0.4465), std: (0.2023, 0.1994, 0.2010)
        
        Returns:
            Callable: The transform, an ``albumentations.Compose`` object.
        """
        return A.Compose(
            [
                A.Normalize(
                    mean=cifar10.mean, std=cifar10.std
                ),  # Normalizing
            ]
        )


class _cifar10_dataset(datasets.CIFAR10):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Callable = None,
        target_transform: Callable = None,
        download: bool = False,
    ):
        super(_cifar10_dataset, self).__init__(
            root, train, transform, target_transform, download
        )

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        img, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            try:
                img = self.transform(image=img)["image"].transpose(2, 1, 0)
            except TypeError:
                # at this stage, assuming that the transform isnt an albumentations transform
                # this error will occur if the transform given does not have an ``image`` keyword argument,
                # or the return type isnt a dict (like albumentations)
                img = self.transform(img).transpose(2, 1, 0)

        if self.target_transform is not None:
            try:
                target = self.target_transform(image=target)["image"].transpose(2, 1, 0)
            except TypeError:
                # at this stage, assuming that the transform isnt an albumentations transform
                # this error will occur if the transform given does not have an ``image`` keyword argument,
                # or the return type isnt a dict (like albumentations)
                target = self.target_transform(target).transpose(2, 1, 0)

        return img, target
