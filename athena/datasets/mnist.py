from typing import Callable, Tuple

import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from .base_dataset import BaseDataset


class mnist(BaseDataset):
    def build(self):
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

    def _default_train_transform(self):
        return A.Compose(
            [
                A.Rotate(limit=5),  # Randomly rotating the image in the range -5,5 degrees
                A.Normalize(mean=(0.1307,), std=(0.3081,)),  # Normalizing
            ]
        )

    def _default_test_transform(self):
        return A.Compose(
            [
                A.Normalize(mean=(0.1307,), std=(0.3081,)),  # Normalizing
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
            img = self.transform(image=img)["image"]
            img = np.expand_dims(img, 0)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
