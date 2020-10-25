from typing import Callable, Tuple

import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from .base import BaseDataset, DataLoaderBuilder
from athena.utils.transforms import ToNumpy, ToTensor


class mnist(datasets.MNIST):

    mean = (0.1307,)  #: mean of the dataset.
    std = (0.3081,)  #: std of the dataset.

    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        transform: Callable = None,
        target_transform: Callable = None,
        download: bool = False,
        use_default_transforms: bool = False,
    ):
        """
        MNIST dataset.

        Args:
            root (str): The root directory of the dataset. Defaults to "./data". 
            train (bool, optional): Whether its train or test dataset. Defaults to ``True``.
            transform (Callable, optional): The tranform to apply on the data. Defaults to ``None``.
            target_transform (Callable, optional): The transform to apply on the labels. Defaults \
                to ``None``.
            download (bool, optional): Whether the dataset should be downloaded or not. Defaults \
                to ``False``.
            use_default_transforms (bool, optional): Whether the default transforms must be used \
                or not. Defaults to ``False``.
        """
        super(mnist, self).__init__(root, train, transform, target_transform, download)
        self.use_default_transforms = use_default_transforms
        self._set_transforms()

    def __getitem__(self, index) -> Tuple[np.ndarray, int]:
        img, target = self.data[index].numpy(), int(self.targets[index])

        # converting uint8 numpy array (values range from 0-255)
        # to float32 array (values ranging from 0-1)
        img = img.astype(np.float32) / 255
        # adding channel dimension at the end
        img = img[:, :, None]  # final shape: (H, W, C)

        if self.transform is not None:
            if isinstance(
                self.transform, (A.BasicTransform, A.core.composition.BaseCompose)
            ):
                img = self.transform(image=img)["image"]
            else:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

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
                A.Lambda(ToNumpy, name="ToNumpy"),
                A.Rotate(
                    limit=5
                ),  # Randomly rotating the image in the range -5,5 degrees
                A.Normalize(
                    mean=mnist.mean, std=mnist.std, max_pixel_value=1.0
                ),  # Normalizing
                A.Lambda(ToTensor, name="ToTensor"),
            ]
        )

    def default_val_transform(self):
        """
        Default mnist validation transforms. Performs

        * Normalization using mean: (0.1307,) and std: (0.3081,)

        Returns:
            Callable: The transform, an ``albumentations.Compose`` object.
        """
        return A.Compose(
            [
                A.Lambda(ToNumpy, name="ToNumpy"),
                A.Normalize(
                    mean=mnist.mean, std=mnist.std, max_pixel_value=1.0
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
                self.transform = self.default_val_transform()

    @classmethod
    def builder(cls) -> DataLoaderBuilder:
        """
        Returns a DataLoader builder.
        """

        return DataLoaderBuilder(cls)
