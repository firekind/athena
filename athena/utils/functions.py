import math
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from athena import Experiment, ClassificationSolver


def plot_experiments(experiments: List[Experiment], save_path: str = None, figsize: Tuple[int, int] = (18, 13)):
    """
    Plots the train and test losses and accuracies of the given experiments.

    Args:
        experiments (List[Experiment]): The list of ``Experiments`` to plot
        save_path (str, optional): The path to save the plot. Defaults to None.
        figsize (Tuple[int, int], optional): The size of the plot. Defaults to (18, 13).
    """

    train_losses = [exp.history.train_losses for exp in experiments]
    train_accs = [exp.history.train_accs for exp in experiments]
    test_losses = [exp.history.test_losses for exp in experiments]
    test_accs = [exp.history.test_accs for exp in experiments]
    data = [train_losses, train_accs, test_losses, test_accs]
    titles = ["Train loss", "Train accuracy", "Test loss", "Test accuracy"]
    legends = [exp.name for exp in experiments]

    nrows = 2
    ncols = 2
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)

    for i in range(nrows):
        for j in range(ncols):
            index = i * ncols + j
            ax[i, j].set_title(titles[index])

            for k, legend in enumerate(legends):
                ax[i, j].plot(data[index][k], label=legend)

            ax[i, j].legend()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.25)


def plot_misclassified(
    number: int,
    experiment: Experiment,
    data_loader: DataLoader,
    device: str,
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 15)
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
    """

    if not isinstance(experiment.solver, ClassificationSolver):
        raise Exception(
            "Only experiments with a ClassificationSolver can be used to plot misclassified images."
        )

    image_data, predicted, actual = experiment.solver.get_misclassified(
        data_loader, device
    )
    nrows = math.floor(math.sqrt(number))
    ncols = math.ceil(math.sqrt(number))

    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)

    for i in range(nrows):
        for j in range(ncols):
            index = i * ncols + j
            ax[i, j].axis("off")
            ax[i, j].set_title(
                "Predicted: %d\nActual: %d" % (predicted[index], actual[index])
            )
            ax[i, j].imshow(image_data[index].cpu().numpy(), cmap="gray_r")

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.25)
