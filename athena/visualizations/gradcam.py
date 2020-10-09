import math
from typing import Tuple, Callable

import cv2
import matplotlib.pyplot as plt
from matplotlib import axes
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from athena.utils.transforms import UnNormalize
from .utils import plot_grid
from athena.utils import get_misclassified


class GradCam:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Performs Gradient-weighted Class Activation Mapping algorithm on the model and
        using the target layer. Code taken from
        `here <https://github.com/vickyliin/gradcam_plus_plus-pytorch>`_.

        Args:
            model (nn.Module): The model to apply on.
            target_layer (nn.Module): The layer of the model to apply on.
        """

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

        # setting hooks, so that the gradients and activations
        # of the target layer are captured.
        self.target_layer.register_backward_hook(backward_hook)
        self.target_layer.register_forward_hook(forward_hook)

    def forward(
        self, x: torch.Tensor, class_idx: int = None, retain_graph: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        applies the gradcam algorithm with given input and class.

        Args:
            x (torch.Tensor): The input image. Should have shape ``(C, H, W)``
            class_idx (int, optional): The index of the class to weight against. If \
                None, uses the class with the highest activation. Defaults to None.
            retain_graph (bool, optional): Whether to retain the computational graph or \
                not. Look at the pytorch documentation for more info regarding this \
                parameter. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the heatmap mask (shape: ``(H, W)``) \
                and the output of forward proping the given input ``x`` in the model.
        """

        assert x.ndim == 3, "Input needs to be 3 Dimensional"

        # adding batch dimension
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


class GradCamPP(GradCam):
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Performs GradCam Plus Plus algorithm on the model and 
        using the target layer.
        Code taken from `here <https://github.com/vickyliin/gradcam_plus_plus-pytorch>`_.

        Args:
            model (nn.Module): The model to apply on.
            target_layer (nn.Module): The layer of the model to apply on.
        """
        super(GradCamPP, self).__init__(model, target_layer)

    def forward(
        self, x: torch.Tensor, class_idx: int = None, retain_graph: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        applies the gradcam plus plus algorithm with given input and class.

        Args:
            x (torch.Tensor): The input image. Should have shape ``(C, H, W)``
            class_idx (int, optional): The index of the class to weight against. If \
                None, uses the class with the highest activation. Defaults to None.
            retain_graph (bool, optional): Whether to retain the computational graph or \
                not. Look at the pytorch documentation for more info regarding this \
                parameter. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the heatmap mask (shape: ``(H, W)``) \
                and the output of forward proping the given input ``x`` in the model.
        """
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

        # calculating the numerator according to the formula
        numerator = gradient.pow(2)  # shape: (1, C, H, W)

        # calculating the denominator
        denominator = (2 * numerator) + (
            (activation * gradient.pow(3))
            .sum((-1, -2))  # summing along H and W dimensions
            .view(b, c, 1, 1)
        )
        # replacing all the 0's in the denominator with 1 (to avoid 0 divison error)
        denominator = torch.where(
            denominator != 0, denominator, torch.ones_like(denominator)
        )

        # calculating alpha
        alpha = numerator / (denominator + 1e-7)

        # calculating weights
        weights = (
            (alpha * F.relu(score.exp() * gradient)).sum((-1, -2)).view(b, c, 1, 1)
        )

        # calculating heatmap mask
        mask = (weights * activation).sum(
            dim=1, keepdim=True
        )  # summing along channel dimension. shape: (1, H, W)
        mask = F.relu(mask)  # performing relu to remove negative values

        # upsampling the heat map to match input image size
        mask = F.upsample(mask, size=(h, w), mode="bilinear", align_corners=False)

        # normalize
        mask = (mask - mask.min()) / (mask.max() - mask.min())

        # getting rid of unecessary dimensions
        mask = mask.squeeze()  # shape: (H, W)

        # setting model to training mode if it wasn't already in eval mode
        # (if it was already in eval mode then caller set it to eval mode,
        # so don't change the mode)
        if not already_in_eval:
            self.model.train()

        return mask, logits


def overlay_gradcam_mask(
    mask: torch.Tensor, image: torch.Tensor, opacity: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Overlays the heatmap mask on the image, and also converts the heatmap mask
    to an actual heatmap.

    Args:
        mask (torch.Tensor): The mask. Should have shape as ``(H, W)``
        image (torch.Tensor): The image to apply the mask on. Should have shape as ``(C, H, W)``
        opacity (float, optional): The amount of opacity to apply to the heatmap mask. Defaults to 1.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The heatmap and the resultant image. Shape of 
        both images is ``(C, H, W)``.
    """
    # converting mask to numpy int type to apply cv2 functions
    heatmap = (255 * mask).type(torch.uint8).cpu().numpy()

    # applying colormap on converted mask
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # converting to float tensor
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)

    # converting bgr to rgb (cuz of opencv) and applying transparency
    heatmap = heatmap[[2, 1, 0], :, :] * opacity

    # overlaying the heatmap with input image
    result = heatmap + image.cpu()

    # scaling the result
    result = result / result.max()

    return heatmap, result


def apply_gradcam(
    module: pl.LightningModule,
    target_layer: nn.Module,
    image: torch.Tensor,
    transform: Callable = None,
    class_idx: int = None,
    retain_graph: bool = False,
    device: str = "cpu",
    use_gradcampp: bool = True,
    alpha: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the gradcam (or gradcam plus plus) algorithm using a model and target layer on an image, and
    returns the result.

    Args:
        module (pl.LightningModule): The model to use.
        target_layer (nn.Module): The layer the gradcam algorithm should be applied on.
        image (torch.Tensor): The image to use. SHape: ``(C, H, W)``
        transform (Callable, optional): The transform to apply on the image before sending it\
            to the model. Defaults to None.
        class_idx (int, optional): The index of the class to using in the algorithm. If ``None``,\
            uses the class with the highest activation. Defaults to None.
        retain_graph (bool, optional): Whether to retain the computational graph or not. Look at \
            the pytorch documentations for info about this parameter. Defaults to False.
        device (str, optional): A valid pytorch device string. Defaults to "cpu".
        use_gradcampp (bool, optional): Whether to use the gradcam plus plus algorithm or the \
            gradcam algorithm. Defaults to True.
        alpha (float, optional): The amount of opacity to apply to the heatmap mask. Defaults to 1.0.


    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The overlaid image and the generated heatmap.
    """

    assert image.ndim == 3, "Input image should be a 3 dimensional array"

    # creating the gradcam object
    if use_gradcampp:
        cam = GradCamPP(module.to(device), target_layer.to(device))
    else:
        cam = GradCam(module.to(device), target_layer.to(device))

    # transforming image
    transformed_image = transform(image) if transform is not None else image
    transformed_image = transformed_image.to(device)

    # generating mask
    mask, logits = cam(transformed_image, class_idx, retain_graph)

    # generating overlay
    heatmap, overlaid = overlay_gradcam_mask(mask, image, alpha)

    return heatmap, overlaid


def plot_gradcam(
    module: pl.LightningModule,
    target_layer: nn.Module,
    image: torch.Tensor,
    transform: Callable = None,
    class_idx: int = None,
    retain_graph: bool = False,
    device: str = "cpu",
    use_gradcampp: bool = True,
    alpha: float = 1.0,
    figsize: Tuple[int, int] = (10, 15),
    save_path: str = None,
):
    """
    Applies gradcam and plots it on a matplotlib figure.

    Args:
        module (pl.LightningModule): The model to use.
        target_layer (nn.Module): The layer the gradcam algorithm should be applied on.
        image (torch.Tensor): The image to use. SHape: ``(C, H, W)``
        transform (Callable, optional): The transform to apply on the image before sending it\
            to the model. Defaults to None.
        class_idx (int, optional): The index of the class to using in the algorithm. If ``None``,\
            uses the class with the highest activation. Defaults to None.
        retain_graph (bool, optional): Whether to retain the computational graph or not. Look at \
            the pytorch documentations for info about this parameter. Defaults to False.
        device (str, optional): A valid pytorch device string. Defaults to "cpu".
        use_gradcampp (bool, optional): Whether to use the gradcam plus plus algorithm or the \
            gradcam algorithm. Defaults to True.
        alpha (float, optional): The amount of opacity to apply to the heatmap mask. Defaults to 1.0.
        figsize (Tuple[int, int], optional): The size of the matplotlib figure. Defaults to ``(10, 15)``.
        save_path (str, optional): The path to save the matplotlib figure. Defaults to ``None``.
    """
    # generating overlaid image
    _, overlaid = apply_gradcam(
        module,
        target_layer,
        image,
        transform,
        class_idx,
        retain_graph,
        device,
        use_gradcampp,
        alpha
    )

    # clipping input range
    if torch.is_floating_point(overlaid):
        overlaid = torch.clamp(overlaid, 0, 1)
    else:
        overlaid = torch.clamp(overlaid, 0, 255)

    # creating figure
    fig, ax = plt.subplots(figsize=figsize)

    # turning of axis lines
    ax.axis("off")

    # drawing image
    ax.imshow(overlaid.permute(1, 2, 0).cpu().numpy())

    # saving model if save path is provided
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.25)


def gradcam_misclassified(
    number: int,
    experiment: "Experiment",
    target_layer: nn.Module,
    dataloader: DataLoader,
    device: str,
    class_labels: Tuple = None,
    figsize: Tuple[int, int] = (10, 15),
    save_path: str = None,
    mean: Tuple = None,
    std: Tuple = None,
    opacity: float = 1.0
):
    """
    Plots gradcam to the misclassified images of the experiment.

    Args:
        number (int): The number of misclassified images on which gradcam should be applied.
        experiment (Experiment): The experiment to use.
        target_layer (nn.Module): The target layer for gradcam.
        dataloader (DataLoader): The dataloader to use.
        device (str): A valid pytorch device string.
        class_labels (Tuple, optional): The class labels. Defaults to None.
        figsize (Tuple[int, int], optional): Size of the plot. Defaults to (10, 15).
        save_path (str, optional): Path to save the plot. Defaults to None.
        mean (Tuple, optional): The mean of the dataset. If given, image will be unnormalized using \
            this before overlaying. Defaults to None.
        std (Tuple, optional): The std of the dataset. If given, image will be unnormalized using \
            this before overlaying. Defaults to None.
        opacity (float, optional): The amount of opacity to apply to the heatmap mask. Defaults to 1.0.
    """

    # getting data of the misclassified images
    image_data, predicted, actual = get_misclassified(
        experiment.get_solver(), dataloader, device
    )

    # generating the array to store the misclassified images
    overlays = torch.zeros(
        number, image_data.size(1), image_data.size(2), image_data.size(3)
    )

    # creating the unnormalizer if needed
    unnormalize = None if mean is None and std is None else UnNormalize(mean, std)

    # initializing gradcam
    cam = GradCamPP(experiment.get_model().to(device), target_layer.to(device))

    # for every image in the misclassified images
    for idx, image in enumerate(image_data):
        if idx >= number:
            break

        # generate heatmap mask
        heatmap_mask, _ = cam(
            image.to(device),
            class_idx=predicted[idx],
        )

        # generate overlay
        overlays[idx] = overlay_gradcam_mask(
            heatmap_mask,
            image if unnormalize is None else unnormalize(image),
            opacity
        )[1]

    # plot results
    plot_grid(
        number,
        lambda idx, ax: _plot_image(
            overlays[idx], predicted[idx], actual[idx], class_labels, ax
        ),
        figsize,
        save_path,
    )


def _plot_image(
    image_data: torch.Tensor,
    predicted: torch.Tensor,
    actual: torch.Tensor,
    class_labels: torch.Tensor,
    ax: axes.Axes,
):
    """
    Plots an image.

    Args:
        image_data (torch.Tensor): The image data.
        predicted (torch.Tensor): The predicted class
        actual (torch.Tensor): The actual class
        class_labels (torch.Tensor): The class labels
        ax (axes.Axes): The axes to plot on.
    """

    # turning off the axis lines in the plot
    ax.axis("off")

    # setting title
    ax.set_title(
        "Predicted: %s\nActual: %s"
        % (
            int(predicted) if class_labels is None else class_labels[predicted],
            int(actual) if class_labels is None else class_labels[actual],
        )
    )

    # clipping input range
    if torch.is_floating_point(image_data):
        image_data = torch.clamp(image_data, 0, 1)
    else:
        image_data = torch.clamp(image_data, 0, 255)

    # plotting image
    ax.imshow(image_data.permute(1, 2, 0).cpu().numpy())