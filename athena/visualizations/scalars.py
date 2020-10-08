import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib import axes
from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer

from .utils import plot_grid


def plot_scalars(
    log_dir: str, figsize: Tuple[int, int] = (15, 10), save_path: str = None
):
    """
    Plots the scalars from a tensorboard log directory as a single matplotlib
    plot.

    Args:
        log_dir (str): The path to the tensorboard log directory.
        figsize (Tuple[int, int]): The size of the figure. Defaults to ``(15, 10)``.
        save_path (str, optional): The path to save the figure to. Defaults to None.
    """

    # creating ``EventMultiplexer`` to read tensorboard log files
    mux = EventMultiplexer().AddRunsFromDirectory(log_dir)

    # reading log files
    mux.Reload()

    # getting list of scalars from all the log files
    scalars = list(
        {
            metric
            for run in mux.Runs().keys()
            for metric in mux.GetAccumulator(run).Tags()["scalars"]
        }
    )
    scalars.sort()

    # getting the list of runs
    runs = {}
    for run in mux.Runs().keys():
        runs[run] = {}
        runs[run]["label"] = run
        runs[run]["run"] = run

    if len(runs) == 1 and list(mux.Runs().keys())[0] == ".":
        runs["."]["label"] = os.path.normpath(log_dir).split(os.sep)[-1]

    plot_grid(
        len(scalars),
        lambda idx, ax: _plot_scalar(mux, scalars[idx], runs, ax),
        figsize=(15, 10),
        save_path=save_path,
    )


def _plot_scalar(mux: EventMultiplexer, scalar: str, runs: Dict, ax: axes.Axes):
    """
    Plots a scalar for the given runs.

    Args:
        scalar (str): The metric to plot.
        runs (Dict): The runs to plot.
        ax (axes.Axes): The ``~matplotlib.axes.Axes`` to plot on.
    """

    # setting title of the plot
    ax.set_title(scalar)

    # for every run
    for run in runs:
        # plot the scalar of the run
        _plot_run(mux, scalar, runs[run], ax)

    # enable legend
    ax.legend()


def _plot_run(mux: EventMultiplexer, scalar: str, run: Dict, ax: axes.Axes):
    """
    Plots a scalar of a given run.

    Args:
        scalar (str): The scalar to plot.
        run (str): The run to plot.
        ax (axes.Axes): The ``~matplotlib.axes.Axes`` to plot on.
    """

    accumulator = mux.GetAccumulator(run["run"])

    # if the run has not recorded that scalar
    if scalar not in accumulator.Tags()["scalars"]:
        return

    _, steps, data = zip(*accumulator.Scalars(scalar))

    # plot the scalar
    ax.plot(steps, data, label=run["label"])
