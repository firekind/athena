from typing import Callable, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from athena.utils.transforms import ToNumpy, ToTensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .base_dataset import BaseDataset


class cifar10(BaseDataset):

    mean = (0.4914, 0.4822, 0.4465)  #: mean of the dataset.
    std = (0.2023, 0.1994, 0.2010)  #: std of the dataset.

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
        super(cifar10, self).create()

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
        * Random crop of 32x32 (after padding it by 4 on each side)
        * Random horizontal flip
        * 8x8 Cutout

        Returns:
            Callable: The transform, an ``albumentations.Compose`` object.
        """
        return A.Compose(
            [
                A.Lambda(ToNumpy),
                A.Normalize(
                    mean=datasets.cifar10.mean,
                    std=datasets.cifar10.std,
                    max_pixel_value=1.0,
                ),
                A.PadIfNeeded(40, 40, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.RandomCrop(32, 32),
                A.HorizontalFlip(),
                A.Cutout(num_holes=1),
                A.Lambda(ToTensor),
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
                A.Lambda(ToNumpy),
                A.Normalize(
                    mean=cifar10.mean, std=cifar10.std, max_pixel_value=1.0
                ),  # Normalizing
                A.Lambda(ToTensor),
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

        # channels first cuz this is used in solver while writing the model
        # to tensorboard
        self.input_shape = (3, 32, 32)

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        img, target = self.data[index], int(self.targets[index])  # img shape: (H, W, C)

        # converting img from a uint8 np array (with range 0-255) to float32 np array (with range 0-1)
        # and channels last.
        img = img.astype(np.float32) / 255

        if self.transform is not None:
            if isinstance(
                self.transform, (A.BasicTransform, A.core.composition.BaseCompose)
            ):
                img = self.transform(image=img)["image"]
            else:
                img = self.transform(img)

        if self.target_transform is not None:
            if isinstance(
                self.target_transform,
                (A.BasicTransform, A.core.composition.BaseCompose),
            ):
                target = self.target_transform(image=target)["image"]
            else:
                target = self.target_transform(target)

        return img, target
