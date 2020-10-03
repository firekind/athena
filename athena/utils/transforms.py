from typing import List

import torch
import numpy as np


def ToNumpy(x: torch.Tensor, **kwargs) -> np.ndarray:
    """
    Function that converts a torch tensor to a numpy array. Valid for tensors with shape ``(C, H, W)``
    or ``(H, W)``

    Args:
        x (torch.Tensor): The input to convert. If the input is a ``np.ndarray``, it doesn't \
            do anything.

    Raises:
        ValueError: If the type of the input is not ``torch.Tensor`` or a ``np.ndarray``

    Returns:
        np.ndarray: A ``np.ndarray`` with shape ``(H, W, C)``.
    """

    if not isinstance(x, (torch.Tensor, np.ndarray)):
        raise ValueError(
            "Expected numpy array or torch.Tensor, but got {}".format(type(x))
        )

    if isinstance(x, np.ndarray):
        return x

    if x.ndim == 2:
        x = x.unsqueeze(dim=0)

    return x.permute(1, 2, 0).cpu().numpy()


def ToTensor(x: np.ndarray, **kwargs) -> torch.Tensor:
    """
    Function that converts a numpy array of shape ``(H, W, C)`` or ``(H, W)`` to a ``torch.Tensor`` 
    of shape ``(C, H, W)``.

    Args:
        x (np.ndarray): The input to convert.

    Raises:
        ValueError: If the input is not of type ``np.ndarray``.

    Returns:
        torch.Tensor: A ``torch.Tensor`` with shape ``(C, H, W)``.
    """

    if not isinstance(x, np.ndarray):
        raise ValueError("Expected np.ndarray, but got {}".format(type(x)))
    
    if x.ndim == 2:
        x = x[:, :, None]
    
    # code from torchvision.functional.to_tensor
    x = torch.tensor(x.transpose(2, 0, 1)).contiguous()

    if isinstance(x, torch.ByteTensor):
        return x.float().div(255)
    return x


class UnNormalize:
    def __init__(self, mean: List, std: List):
        """
        Performs the reverse of a normalization operation.

        Args:
            mean (List): The mean.
            std (List): The std.
        """
        self.mean = mean
        self.std = std
    
    def __call__(self, x):
        tensor = x.clone()
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)

        return tensor