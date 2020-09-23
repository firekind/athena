import math
from typing import Tuple, Callable

import cv2
import matplotlib.pyplot as plt
from matplotlib import axes
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class GradCam:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        # defining variables where the calculated gradients and
        # activations of the target layer will eventually be
        # stored.
        self.gradients: torch.Tensor = None
        self.activations: torch.Tensor = None

        # defining hooks
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # setting hooks
        self.target_layer.register_backward_hook(backward_hook)
        self.target_layer.register_forward_hook(forward_hook)

    def forward(
        self, x: torch.Tensor, class_idx: int = None, retain_graph: bool = False
    ) -> torch.Tensor:
        assert x.ndim == 3, "Input needs to be 3 Dimensional"

        x = x.unsqueeze(dim=0)
        b, c, h, w = x.shape

        # flag to check if model is already in eval mode
        already_in_eval: bool = not self.model.training

        # setting model to eval mode
        self.model.eval()

        # forward prop
        logits = self.model(x)

        # getting the class with respect to which gradients
        # should be calculated
        if class_idx is None:
            score = logits[:, logits.argmax(dim=1)].squeeze()
        else:
            score = logits[:, class_idx].squeeze()

        # zeroing out any leftover gradients in the model
        self.model.zero_grad()

        # calculating the gradients (dy_c/dA_k)
        # (importance of feature map k for a target class c)
        score.backward(retain_graph=retain_graph)

        # extracting the gradient
        gradient = self.gradients
        activation = self.activations
        b, c, _, _ = gradient.size()

        # calculating neuron importance weights (alpha)
        alpha = gradient.mean(
            dim=(-1, -2)
        ).view(  # calculating mean along H and W dimensions
            b, c, 1, 1
        )

        # calculating heatmap mask
        mask = (alpha * activation).sum(
            dim=1, keepdim=True
        )  # summing along channel dimension. shape: (1, H, W)
        mask = F.relu(mask)  # performing relu to remove negative values

        # upscaling the heat map to match input image size
        mask = F.upsample(mask, size=(h, w), mode="bilinear", align_corners=False)

        # normalizing the heat map
        mask = (mask - mask.min()) / (mask.max() - mask.min())

        # getting rid of unecessary dimensions
        mask = mask.squeeze()  # shape: (H, W)

        # setting model to training mode if it wasn't already in eval mode
        # (if it was already in eval mode then caller set it to eval mode,
        # so don't change the mode)
        if not already_in_eval:
            self.model.train()

        return mask, logits

    def __call__(
        self, x: torch.Tensor, class_idx: int = None, retain_graph: bool = False
    ):
        return self.forward(x, class_idx, retain_graph)


def overlay_gradcam_mask(
    mask: torch.Tensor, image: torch.Tensor, alpha: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the heatmap mask onto the image.

    Args:
        mask (torch.Tensor): The mask. Should have shape as (H, W)
        image (torch.Tensor): The image to apply the mask on. Should have shape as (C, H, W)
        alpha (float, optional): The amount of transparency to apply to the heatmap mask. Defaults to 1.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The heatmap and the resultant image.
    """
    # converting mask to numpy int type to apply cv2 functions
    heatmap = (255 * mask).type(torch.uint8).cpu().numpy()

    # applying colormap on converted mask
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # converting to float tensor
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)

    # converting bgr to rgb and applying transparency
    heatmap = heatmap[[2, 1, 0], :, :] * alpha

    # overlaying the heatmap with input image
    result = heatmap + image.cpu()

    # scaling the result
    result = result / result.max()

    return heatmap, result


def apply_gradcam(
    model: nn.Module,
    target_layer: nn.Module,
    image: torch.Tensor,
    transform: Callable = None,
    class_idx: int = None,
    retain_graph: bool = False,
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 15),
    device: str = "cpu",
):
    assert image.ndim == 3, "Input image should be a 3 dimensional array"

    # creating the gradcam object
    cam = GradCam(model.to(device), target_layer.to(device))

    # transforming image
    transformed_image = transform(image) if transform is not None else image
    transformed_image = transformed_image.to(device) 

    # generating mask
    mask, logits = cam(transformed_image, class_idx, retain_graph)

    # generating overlay
    heatmap, overlaid = overlay_gradcam_mask(mask, image)

    # clipping input range
    if torch.is_floating_point(overlaid):
        overlaid = torch.clamp(overlaid, 0, 1)
    else:
        overlaid = torch.clamp(overlaid, 0, 255)

    # creating figure
    fig, ax = plt.subplots()

    # turning of axis lines
    ax.axis("off")

    # drawing image
    ax.imshow(overlaid.permute(1, 2, 0).cpu().numpy())

    # saving model if save path is provided
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.25)
