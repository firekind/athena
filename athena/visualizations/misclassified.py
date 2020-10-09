import math
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib import axes
import torch
from torch.utils.data import DataLoader

from athena.solvers import ClassificationSolver
from athena.utils.transforms import UnNormalize
from .utils import plot_grid
from athena.utils import get_misclassified


def plot_misclassified(
    number: int,
    experiment: "Experiment",
    data_loader: DataLoader,
    device: str = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 15),
    cmap: str = "gray_r",
    class_labels: Tuple[str] = None,
    mean: Tuple = None,
    std: Tuple = None,
):
    """
    Plots the misclassified images.

    Args:
        number (int): The number of misclassified images to plot.
        experiment (Experiment): The experiment that should be used to get the misclassified images
        data_loader (DataLoader): The ``DataLoader`` of the input data
        device (str, optional): A valid pytorch device string. Defaults to None. If None, will use \
            the ``experiment``'s device.
        save_path (str, optional): The path to save the plot. Defaults to None.
        figsize (Tuple[int, int], optional): The size of the plot. Defaults to (10, 15).
        cmap (str, optional): The cmap to use while plotting. Defaults to 'gray_r'
        class_labels (Tuple[str], optional): The class labels to use. Defaults to None.
        mean (Tuple, optional): The mean of the dataset. If given, image will be unnormalized using \
            this before overlaying. Defaults to None.
        std (Tuple, optional): The std of the dataset. If given, image will be unnormalized using \
            this before overlaying. Defaults to None.
    
    Raises:
        Exception: When solver of type :class:`athena.solvers.classification_solver.ClassificationSolver` is not being used, \
            and when the number of misclassified images are less than the number of images being requested to plot.
    """

    if not isinstance(experiment.get_solver(), ClassificationSolver):
        raise Exception(
            "Only experiments with a ClassificationSolver can be used to plot misclassified images."
        )

    if mean is not None and std is not None:
        unorm = UnNormalize(mean, std)
    else:
        unorm = None

    # getting the misclassified images by forward proping the model
    image_data, predicted, actual = get_misclassified(
        experiment.get_solver(), data_loader, device
    )

    if image_data.ndim == 3:
        image_data = image_data.unsqueeze(0)

    # checking if number of misclassified images < number of images to plot
    if len(image_data) < number:
        raise Exception(
            "Number of misclassified images are less than the number of images requested to plot."
        )

    plot_grid(
        number,
        lambda idx, ax: _plot_image(
            image_data[idx] if unorm is None else unorm(image_data[idx]),
            predicted[idx],
            actual[idx],
            class_labels,
            ax,
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
    ax.imshow(image_data.permute(1, 2, 0).cpu().numpy(), cmap="gray_r")
