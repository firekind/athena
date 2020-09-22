import math
from typing import List, Tuple, Union

from matplotlib import axes
import matplotlib.pyplot as plt

from athena import Experiment, Experiments


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
    if not experiment.get_solver().get_history().has_metric(metric):
        return

    # plot the metric
    ax.plot(
        experiment.get_solver().get_history().get_metric(metric), label=experiment.name
    )
