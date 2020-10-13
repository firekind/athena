import math
from typing import Tuple, Callable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def plot_grid(
    number: int,
    plot_fn: Callable[[int, Axes], None],
    figsize: Tuple[int, int] = (10, 15),
    save_path: str = None,
):
    """
    Plots images in a grid.

    Args:
        number (int): The number of images to plot.
        plot_fn (Callable[[int, Axes], None]): The function to be called for each plot in the grid.\
            The 1D index of the plot and the `matplotlib.axes.Axes` will be passed as arguments.\
            Typically this function actually plots the graph / image using ``plt.plot``, \
            ``plt.imshow``, etc.
        figsize (Tuple[int, int], optional): The size of the figure. Defaults to (10, 15).
        save_path (str, optional): The path to save the figure. Defaults to None.
    """
    # calculating the number of rows and columns in the plot
    nrows = math.floor(math.sqrt(number))
    ncols = math.ceil(number / nrows)

    # creating the empty plot
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)

    # if there is only one image to plot
    if nrows == 1 and ncols == 1:
        plot_fn(0, ax)

    # if there is only one row of images to plot
    elif nrows == 1:
        for i in range(len(ax)):
            plot_fn(i, ax[i])

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
                plot_fn(index, ax[i, j])

            if index >= number:
                break

    # saving model if save path is provided
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.25)
