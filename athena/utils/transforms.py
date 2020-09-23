import torch
import numpy as np


def ToNumpy(x, **kwargs):
    if not isinstance(x, (torch.Tensor, np.ndarray)):
        raise ValueError(
            "Expected numpy array or torch.Tensor, but got {}".format(type(x))
        )

    if isinstance(x, np.ndarray):
        return x

    return x.permute(1, 2, 0).cpu().numpy()


def ToTensor(x, **kwargs):
    if not isinstance(x, np.ndarray):
        raise ValueError("Expected np.ndarray, but got {}".format(type(x)))
    
    if x.ndim == 2:
        x = x[:, :, None]
    
    # code from torchvision.functional.to_tensor
    x = torch.tensor(x.transpose(2, 0, 1)).contiguous()

    if isinstance(x, torch.ByteTensor):
        return x.float().div(255)
    return x