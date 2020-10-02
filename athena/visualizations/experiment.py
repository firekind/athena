import math
from typing import List, Tuple, Union

from matplotlib import axes
import matplotlib.pyplot as plt

from athena import Experiment, Experiments
from .utils import plot_grid


def plot_experiments(
    experiments: Union[Experiments, List[Experiment]],
    save_path: str = None,
    figsize: Tuple[int, int] = (18, 13),
):
    """
    Plots the metrics of the experiments given. Experiments that have recorded the same metric will have overlapping plots.

    Args:
        experiments (Union[Experiments, List[Experiment]]): The list of ``Experiments`` to plot
        save_path (str, optional): The path to save the plot. Defaults to None.
        figsize (Tuple[int, int], optional): The size of the plot. Defaults to (18, 13).

    Raises:
        Exception: When there are no metrics to plot.
    """
    if isinstance(experiments, Experiments):
        experiments = list(experiments)

    # getting the list of metrics to plot
    metric_names = list(
        {
            metric
            for exp in experiments
            for metric in exp.get_solver().get_history().get_metric_names()
        }
    )
    metric_names.sort()
    num_metrics = len(metric_names)

    plot_grid(
        num_metrics,
        lambda idx, ax: _plot_metric(metric_names[idx], experiments, ax),
        figsize,
        save_path,
    )


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
    if not experiment.get_solver().get_history().has_metric(metric):
        return

    # plot the metric
    ax.plot(
        experiment.get_solver().get_history().get_metric(metric), label=experiment.name
    )
