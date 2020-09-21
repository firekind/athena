from typing import Callable, Tuple

import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from .base_dataset import BaseDataset


class mnist(BaseDataset):

    mean = (0.1307,) #: mean of the dataset.
    std = (0.3081,) #: std of the dataset.

    def __init__(self):
        """
        The mnist dataset.
        """
        super(mnist, self).__init__()

    def build(self) -> DataLoader:
        """
        Builds the dataset and returns a pytorch ``DataLoader``.

        Returns:
            DataLoader: The mnist ``DataLoader``.
        """
        super(mnist, self).build()

        return DataLoader(
            _mnist_dataset(
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

    def default_train_transform(self) -> Callable:
        """
        Default mnist training transforms. Performs

        * Random Rotation between -5 and 5 degrees

        * Normalization using mean: (0.1307,) and std: (0.3081,)
        
        Returns:
            Callable: The transform, an ``albumentations.Compose`` object.
        """
        return A.Compose(
            [
                A.Rotate(limit=5),  # Randomly rotating the image in the range -5,5 degrees
                A.Normalize(mean=mnist.mean, std=mnist.std),  # Normalizing
            ]
        )

    def default_test_transform(self):
        """
        Default mnist test transforms. Performs

        * Normalization using mean: (0.1307,) and std: (0.3081,)
        
        Returns:
            Callable: The transform, an ``albumentations.Compose`` object.
        """
        return A.Compose(
            [
                A.Normalize(mean=mnist.mean, std=mnist.std),  # Normalizing
            ]
        )


class _mnist_dataset(datasets.MNIST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Callable = None,
        target_transform: Callable = None,
        download: bool = False,
    ):
        super(_mnist_dataset, self).__init__(
            root, train, transform, target_transform, download
        )

    def __getitem__(self, index) -> Tuple[np.ndarray, int]:
        img, target = self.data[index].numpy(), int(self.targets[index])
        if self.transform is not None:
            try:
                img = self.transform(image=img)["image"]
            except TypeError:
                # at this stage, assuming that the transform isnt an albumentations transform
                # this error will occur if the transform given does not have an ``image`` keyword argument,
                # or the return type isnt a dict (like albumentations)
                img = self.transform(img)
            
            img = np.expand_dims(img, 0)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
