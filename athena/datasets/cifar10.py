from typing import Callable, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from athena.utils.transforms import ToNumpy, ToTensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .base import BaseDataset, DataLoaderBuilder


class cifar10(datasets.CIFAR10):

    mean = (0.4914, 0.4822, 0.4465)  #: mean of the dataset.
    std = (0.2023, 0.1994, 0.2010)  #: std of the dataset.

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Callable = None,
        target_transform: Callable = None,
        download: bool = False,
        use_default_transforms: bool = False,
    ):
        """
        CIFAR10 Dataset.

        Args:
            root (str): The root directory of the dataset.
            train (bool, optional): Whether its train or test dataset. Defaults to ``True``.
            transform (Callable, optional): The tranform to apply on the data. Defaults to ``None``.
            target_transform (Callable, optional): The transform to apply on the labels. Defaults \
                to ``None``.
            download (bool, optional): Whether the dataset should be downloaded or not. Defaults \
                to ``False``.
            use_default_transforms (bool, optional): Whether the default transforms must be used \
                or not. Defaults to ``False``.
        """
        super(cifar10, self).__init__(
            root, train, transform, target_transform, download
        )
        self.use_default_transforms = use_default_transforms
        self._set_transforms()

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
                A.Lambda(ToNumpy, name="ToNumpy"),
                A.Normalize(
                    mean=cifar10.mean,
                    std=cifar10.std,
                    max_pixel_value=1.0,
                ),
                A.PadIfNeeded(40, 40, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.RandomCrop(32, 32),
                A.HorizontalFlip(),
                A.Cutout(num_holes=1),
                A.Lambda(ToTensor, name="ToTensor"),
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
                A.Lambda(ToNumpy, name="ToNumpy"),
                A.Normalize(
                    mean=cifar10.mean, std=cifar10.std, max_pixel_value=1.0
                ),  # Normalizing
                A.Lambda(ToTensor, name="ToTensor"),
            ]
        )

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
                self.transform = self.default_test_transform()

    @classmethod
    def builder(cls) -> DataLoaderBuilder:
        """
        Returns a DataLoader builder.
        """
        return DataLoaderBuilder(cls)
