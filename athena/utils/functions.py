import math
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from matplotlib import axes
import matplotlib.pyplot as plt

from athena import Experiment, ClassificationSolver


def plot_experiments(
    experiments: List[Experiment],
    save_path: str = None,
    figsize: Tuple[int, int] = (18, 13),
):
    """
    Plots the metrics of the experiments given. Experiments that have recorded the same metric will have overlapping plots.

    Args:
        experiments (List[Experiment]): The list of ``Experiments`` to plot
        save_path (str, optional): The path to save the plot. Defaults to None.
        figsize (Tuple[int, int], optional): The size of the plot. Defaults to (18, 13).

    Raises:
        Exception: When there are no metrics to plot.
    """

    # getting the list of metrics to plot
    metric_names = list(
        {metric for exp in experiments for metric in exp.history.get_metric_names()}
    )
    metric_names.sort()
    num_metrics = len(metric_names)

    # checking if number of metrics to plot is > 0
    if num_metrics == 0:
        raise Exception("No metrics to plot.")

    # calculating the number of rows and column in the plot
    nrows = math.floor(math.sqrt(num_metrics))
    ncols = math.ceil(num_metrics / nrows)

    # creating the empty plot
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)

    # if there is only one metric to plot
    if nrows == 1 and ncols == 1:
        _plot_metric(metric_names[0], experiments, ax)

    # if there is only one row of metrics to plot
    elif nrows == 1:
        # for every plot,
        for i in range(len(ax)):
            _plot_metric(metric_names[i], experiments, ax[i])

    # if there is more than one row of metrics to plot
    else:
        for i in range(nrows):
            metric_idx = 0

            for j in range(ncols):

                # coverting i and j values into a single value that can be used
                # to index a 1 dimensional array
                metric_idx = i * ncols + j

                # if the converted index is greater than the number of metrics to plot,
                # that means there are no more metrics to plot, so break
                if metric_idx >= num_metrics:
                    break

                # plot the metric for all the experiments
                _plot_metric(metric_names[metric_idx], experiments, ax[i, j])

            if metric_idx == num_metrics:
                break

    # saving the plot if path is provided
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.25)


def _plot_metric(metric: str, experiments: List[Experiment], ax: axes.Axes):
    """
    Plots a metric for the given list of experiments.

    Args:
        metric (str): The metric to plot.
        experiments (List[Experiment]): The list of experiments whose metrics have to be plotted.
        ax (axes.Axes): The ``~matplotlib.axes.Axes`` to plot on.
    """

    # setting title of the plot
    ax.set_title(metric)

    # for every experiment
    for exp in experiments:
        # plot the metric of the experiment
        _plot_experiment(metric, exp, ax)

    # enable legend
    ax.legend()


def _plot_experiment(metric: str, experiment: Experiment, ax: axes.Axes):
    """
    Plots a metric of a given experiment.

    Args:
        metric (str): The metric to plot.
        experiment (Experiment): The experiment to plot.
        ax (axes.Axes): The ``~matplotlib.axes.Axes`` to plot on.
    """

    # if the experiment has not recorded that metric
    if not experiment.history.has_metric(metric):
        return

    # plot the metric
    ax.plot(experiment.history.get_metric(metric), label=experiment.name)


def plot_misclassified(
    number: int,
    experiment: Experiment,
    data_loader: DataLoader,
    device: str,
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 15),
    cmap: str = "gray_r",
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
    
    Raises:
        Exception: When solver of type :class:`athena.solvers.classification_solver.ClassificationSolver` is not being used, \
            and when the number of misclassified images are less than the number of images being requested to plot.
    """

    if not isinstance(experiment.solver, ClassificationSolver):
        raise Exception(
            "Only experiments with a ClassificationSolver can be used to plot misclassified images."
        )

    # getting the misclassified images by forward proping the model
    image_data, predicted, actual = experiment.solver.get_misclassified(
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
        _plot_image(image_data[0], predicted[0], actual[0], ax)

    # if there is only one row of images to plot
    elif nrows == 1:
        for i in range(len(ax)):
            _plot_image(image_data[i], predicted[i], actual[i], ax[i])

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
                    image_data[index], predicted[index], actual[index], ax[i, j]
                )

            if index >= number:
                break

    # saving model if save path is provided
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.25)


def _plot_image(image_data: torch.Tensor, predicted: int, actual: int, ax: axes.Axes):
    """
    Plots an image.

    Args:
        image_data (torch.Tensor): The image data.
        predicted (int): The class which the model predicted the image belongs to.
        actual (int): The actual class the image belongs to.
        ax (axes.Axes): The ``~matplotlib.axes.Axes`` to plot on.
    """

    # turning off the axis lines in the plot
    ax.axis("off")

    # setting title
    ax.set_title("Predicted: %d\nActual: %d" % (predicted, actual))

    # plotting image
    ax.imshow(image_data.cpu().numpy(), cmap="gray_r")
