import os

import numpy as np
import cv2
import albumentations as A
from .base import BaseDataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.folder import make_dataset, default_loader
from athena.utils import ToNumpy, ToTensor


class tinyimagenet(BaseDataset):

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"  #: url of the dataset.
    mean = (0.4802, 0.4481, 0.3975)  #: mean of the dataset.
    std = (0.2296, 0.2263, 0.2255)  #: std of the dataset.

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        use_default_transforms=False,
    ):
        """
        Tiny ImageNet Dataset.

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
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
            use_default_transforms=use_default_transforms,
        )

        if download:
            self._download_dataset()

        self.dir = os.path.join(root, "tiny-imagenet-200", "train" if train else "val")
        self.class_to_idx = self._load_meta()
        self.samples = self._load_train_samples() if train else self._load_val_samples()
        self.loader = default_loader

    def __getitem__(self, index):
        path: str
        target: int
        path, target = self.samples[index]

        img: np.ndarray = (
            np.array(self.loader(path), dtype=np.float32) / 255
        )  # shape: (H, W, C)

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

    def __len__(self):
        return len(self.samples)

    def _load_meta(self):
        classes = []

        with open(os.path.join(self.root, "tiny-imagenet-200", "wnids.txt"), "r") as f:
            for i, line in enumerate(f):
                classes.append(line.rstrip())

        classes.sort()

        return {class_name: idx for idx, class_name in enumerate(classes)}

    def _get_val_meta(self):
        if self.train:
            raise RuntimeError(
                "Cannot load validation metadata when dataset has train data."
            )

        img_to_class = {}
        with open(os.path.join(self.dir, "val_annotations.txt"), "r") as f:
            for line in f:
                parts = line.rstrip().split("\t")
                img_to_class[parts[0]] = parts[1]

        return img_to_class

    def _load_val_samples(self):
        img_to_class = self._get_val_meta()
        path = os.path.join(self.dir, "images")

        image_files = []
        for f_name in os.listdir(path):
            image_files.append(f_name)

        return [
            (os.path.join(path, f_name), self.class_to_idx[img_to_class[f_name]])
            for f_name in image_files
        ]

    def _load_train_samples(self):
        return make_dataset(self.dir, self.class_to_idx, extensions=("jpeg",))

    def _download_dataset(self):
        download_and_extract_archive(tinyimagenet.url, self.root)

    def default_train_transform(self):
        """
        Default cifar10 training transforms. Performs

        * Normalization using mean: (0.4802, 0.4481, 0.3975), std: (0.2296, 0.2263, 0.2255)
        * Random crop of 64x64 (after padding it by 8 on each side)
        * Random horizontal flip
        * 16x16 Cutout

        Returns:
            Callable: The transform, an ``albumentations.Compose`` object.
        """
        return A.Compose(
            [
                A.Lambda(ToNumpy),
                A.Normalize(
                    mean=tinyimagenet.mean,
                    std=tinyimagenet.std,
                    max_pixel_value=1.0,
                ),
                A.PadIfNeeded(80, 80, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.RandomCrop(64, 64),
                A.HorizontalFlip(),
                A.Cutout(num_holes=1, max_h_size=16, max_w_size=16),
                A.Lambda(ToTensor),
            ]
        )

    def default_test_transform(self):
        """
        Default cifar10 test transforms. Performs

        * Normalization using mean: (0.4802, 0.4481, 0.3975), std: (0.2296, 0.2263, 0.2255)

        Returns:
            Callable: The transform, an ``albumentations.Compose`` object.
        """
        return A.Compose(
            [
                A.Lambda(ToNumpy),
                A.Normalize(
                    mean=tinyimagenet.mean, std=tinyimagenet.std, max_pixel_value=1.0
                ),  # Normalizing
                A.Lambda(ToTensor),
            ]
        )
