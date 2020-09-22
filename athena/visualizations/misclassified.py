import math
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib import axes
import torch
from torch.utils.data import DataLoader

from athena import ClassificationSolver, Experiment

def plot_misclassified(
    number: int,
    experiment: Experiment,
    data_loader: DataLoader,
    device: str,
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 15),
    cmap: str = "gray_r",
    class_labels: Tuple[str] = None,
):
    """
    Plots the misclassified images.

    Args:
        number (int): The number of misclassified images to plot.
        experiment (Experiment): The experiment that should be used to get the misclassified images
        data_loader (DataLoader): The ``DataLoader`` of the input data
        device (str): A valid pytorch device string.
        save_path (str, optional): The path to save the plot. Defaults to None.
        figsize (Tuple[int, int], optional): The size of the plot. Defaults to (10, 15).
        cmap (str, optional): The cmap to use while plotting. Defaults to 'gray_r'
        class_labels (Tuple[str], optional): The class labels to use. Defaults to None.
    
    Raises:
        Exception: When solver of type :class:`athena.solvers.classification_solver.ClassificationSolver` is not being used, \
            and when the number of misclassified images are less than the number of images being requested to plot.
    """

    if not isinstance(experiment.get_solver(), ClassificationSolver):
        raise Exception(
            "Only experiments with a ClassificationSolver can be used to plot misclassified images."
        )

    # getting the misclassified images by forward proping the model
    image_data, predicted, actual = experiment.get_solver().get_misclassified(
        data_loader, device
    )

    # checking if number of misclassified images < number of images to plot
    if len(image_data) < number:
        raise Exception(
            "Number of misclassified images are less than the number of images requested to plot."
        )

    # calculating the number of rows and columns in the plot
    nrows = math.floor(math.sqrt(number))
    ncols = math.ceil(number / nrows)

    # creating the empty plot
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)

    # if there is only one image to plot
    if nrows == 1 and ncols == 1:
        _plot_image(image_data[0], predicted[0], actual[0], ax, class_labels)

    # if there is only one row of images to plot
    elif nrows == 1:
        for i in range(len(ax)):
            _plot_image(image_data[i], predicted[i], actual[i], ax[i], class_labels)

    # if there are multiple rows of images to plot
    else:
        for i in range(nrows):
            index = 0

            for j in range(ncols):
                # converting i and j values to 1-D index value
                index = i * ncols + j

                # checking if the converted index is >= the number of images to plot
                if index >= number:
                    break

                # plotting image
                _plot_image(
                    image_data[index],
                    predicted[index],
                    actual[index],
                    ax[i, j],
                    class_labels,
                )

            if index >= number:
                break

    # saving model if save path is provided
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.25)


def _plot_image(
    image_data: torch.Tensor,
    predicted: int,
    actual: int,
    ax: axes.Axes,
    class_labels: Tuple[str] = None,
):
    """
    Plots an image.

    Args:
        image_data (torch.Tensor): The image data.
        predicted (int): The class which the model predicted the image belongs to.
        actual (int): The actual class the image belongs to.
        ax (axes.Axes): The ``~matplotlib.axes.Axes`` to plot on.
        class_labels (Tuple[str], optional): The class labels to use. Defaults to None.
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