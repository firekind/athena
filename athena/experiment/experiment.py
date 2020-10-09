import os
import shutil
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from athena.solvers.base_solver import BaseSolver
from athena.tuning import LRFinder
from athena.utils import ProgbarCallback
from athena.visualizations import (
    gradcam_misclassified,
    plot_misclassified,
    plot_scalars,
)
from matplotlib.axes import Axes
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader


class Experiment:
    def __init__(
        self,
        name: str,
        solver: BaseSolver,
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        log_dir: str,
        trainer_args: Dict[str, Any],
        resume_from_checkpoint: str = None,
        force_restart: bool = False,
    ):
        """
        Bundles information regarding an experiment. An experiment is performed on a model, with
        a ``Solver`` (subclass of :class:`BaseSolver`.).

        Args:
            name (str): The name of the experiment.
            solver (BaseSolver): The solver the experiment is using.
            epochs (int): The number of epochs to train for.
            train_loader (DataLoader): The train ``DataLoader``.
            val_loader (DataLoader): The validation ``DataLoader``.
            log_dir (str): The log directory.
            trainer_args (Dict[str, Any]): Additional arguments to be given to ``pl.Trainer``.
            resume_from_checkpoint (str, optional): The path to the checkpoint to resume training from.
            force_restart (bool, optional): If True, deletes old checkpoints and restarts training from beginning. \
                Defaults to False.
        """

        self.name = name
        self.solver = solver
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_dir = log_dir
        self.lr_finder = None

        if (
            os.path.exists(log_dir)
            and len(os.listdir(log_dir)) != 0
            and not force_restart
        ):
            raise FileExistsError(
                f"The directory '{log_dir}' is not empty. Use the force_restart property to start"
                " training from scratch or specify a checkpoint to resume from."
            )

        if force_restart and os.path.exists(log_dir):
            shutil.rmtree(log_dir)

        if "gpus" not in trainer_args and torch.cuda.is_available():
            trainer_args["gpus"] = 1

        if "callbacks" in trainer_args:
            trainer_args["callbacks"].append(ProgbarCallback())
        else:
            trainer_args["callbacks"] = [ProgbarCallback()]

        tensorboard_logger = TensorBoardLogger(
            Path(log_dir).parent, name="", version=name
        )
        self.trainer = pl.Trainer(
            max_epochs=epochs,
            logger=tensorboard_logger,
            resume_from_checkpoint=resume_from_checkpoint,
            **trainer_args,
        )

    def run(self):
        """
        Runs the experiment. More specifically, calls the ``pl.Trainer.fit`` method.
        """

        print("\033[1m\033[92m=> Running experiment: %s\033[0m" % self.name, flush=True)

        self.trainer.fit(
            self.solver,
            train_dataloader=self.train_loader,
            val_dataloaders=self.val_loader,
        )

    def lr_range_test(
        self,
        criterion: Callable,
        validate: bool = True,
        start_lr: float = None,
        end_lr: float = 1,
        num_iter: int = 100,
        step_mode: str = "exp",
        smooth_f: float = 0.05,
        diverge_th: float = 5,
        accumulation_steps: int = 1,
        non_blocking_transfer: bool = True,
    ):
        """
        Performs the LR range test.

        Args:
            criterion (Callable): The loss function.
            validate (bool, optional): If True, uses the validation dataset specified to validate every \
                step of the range test. Defaults to True.
            start_lr (float, optional): The starting learning rate of the range test. If ``None``, uses the lr from \
                the optimizer. Defaults to None.
            end_lr (float, optional): The final lr. Defaults to 1.
            num_iter (int, optional): The number of iterations that the test should happen for. Defaults to 100.
            step_mode (str, optional): The step mode. Can be one of "exp" or "linear". Defaults to "exp".
            smooth_f (float, optional): the loss smoothing factor within the [0, 1[ interval. Disabled if set to \
                0, otherwise the loss is smoothed using exponential smoothing. Default: 0.05.
            diverge_th (int, optional): the test is stopped when the loss surpasses the threshold: \
                diverge_th * best_loss. Default: 5.
            accumulation_steps (int, optional): steps for gradient accumulation. If it is 1, gradients are not \
                accumulated. Default: 1.
            non_blocking_transfer (bool, optional): when non_blocking_transfer is set, tries to convert/move \
                data to the device asynchronously if possible, e.g., moving CPU Tensors with pinned memory \
                to CUDA devices. Default: True.
        """

        # creating fresh copy of optimizer (without an lr scheduler)
        optimizer = self.solver.optimizer.__class__(
            self.model.parameters(), **self.solver.optimizer.__dict__["defaults"]
        )

        self.lr_finder = LRFinder(
            self.model,
            optimizer,
            criterion,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        self.lr_finder.range_test(
            self.train_loader,
            self.val_loader if validate else None,
            start_lr=start_lr,
            end_lr=end_lr,
            num_iter=num_iter,
            step_mode=step_mode,
            smooth_f=smooth_f,
            diverge_th=diverge_th,
            accumulation_steps=accumulation_steps,
            non_blocking_transfer=non_blocking_transfer,
        )
        self.lr_finder.plot()
        self.lr_finder.reset()

    def plot_lr_finder(
        self,
        skip_start: int = 10,
        skip_end: int = 5,
        log_lr: bool = True,
        show_lr: float = None,
        ax: Axes = None,
        suggest_lr: bool = True,
    ) -> Union[Axes, Tuple[Axes, float]]:
        """
        Plots the results of the lr range test.

        Args:
            skip_start (int, optional): How many readings to skip from the beginning. Defaults to 10.
            skip_end (int, optional): How many reading to skip from the end. Defaults to 5.
            log_lr (bool, optional): Whether the x-axis should be in log scale or not. Defaults to True.
            show_lr (float, optional): Draws a vertical line at this number. Defaults to None.
            ax (Axes, optional): A matplotlib axes. If specified, figure will be plotted using this. Defaults to None.
            suggest_lr (bool, optional): Whether to suggest an lr or not. Defaults to True.

        Raises:
            ValueError: When an lr range test hasn't been performed.

        Returns:
            Union[Axes, Tuple[Axes, float]]: The matplotlib ``Axes`` and the suggested lr, if any and if specified.
        """

        if self.lr_finder is None:
            raise ValueError("Run a LR range test first.")

        return self.lr_finder.plot(
            skip_start, skip_end, log_lr, show_lr, ax, suggest_lr
        )

    def plot_misclassified(
        self,
        number: int = 25,
        save_path: str = None,
        figsize: Tuple[int, int] = (10, 15),
        class_labels: Tuple = None,
        mean: Tuple = None,
        std: Tuple = None,
    ):
        """
        Plots the images misclassified by the model of this experiment. Will work only
        for classification type solvers.

        Args:
            number (int, optional): The number of misclassified images to plot. Defaults to 25.
            save_path (str, optional): The path to save the plot to. Defaults to None.
            figsize (Tuple[int, int], optional): The size of the figure. Defaults to (10, 15).
            class_labels (Tuple, optional): The class labels, if any. Defaults to None.
            mean (Tuple, optional): The mean of the dataset, used to un-normalize the data \
                before plotting. Defaults to None.
            std (Tuple, optional): The std of the dataset, used to un-normalize the data \
                before plotting. Defaults to None.
        """
        plot_misclassified(
            number,
            self,
            self.val_loader,
            device="cuda" if torch.cuda.is_available() else "cpu",
            save_path=save_path,
            figsize=figsize,
            class_labels=class_labels,
            mean=mean,
            std=std,
        )

    def gradcam_misclassified(
        self,
        target_layer: nn.Module,
        number: int = 25,
        class_labels: Tuple = None,
        figsize: Tuple[int, int] = (10, 15),
        save_path: str = None,
        mean: Tuple = None,
        std: Tuple = None,
        opacity: float = 1.0,
    ):
        """
        Plots gradcam using the misclassified images of the experiment.

        Args:
            target_layer (nn.Module): The target layer for gradcam.
            number (int): The number of misclassified images on which gradcam should be applied.
            class_labels (Tuple, optional): The class labels. Defaults to None.
            figsize (Tuple[int, int], optional): Size of the plot. Defaults to (10, 15).
            save_path (str, optional): Path to save the plot. Defaults to None.
            mean (Tuple, optional): The mean of the dataset. If given, image will be unnormalized using \
                this before overlaying. Defaults to None.
            std (Tuple, optional): The std of the dataset. If given, image will be unnormalized using \
                this before overlaying. Defaults to None.
            opacity (float, optional): The amount of opacity to apply to the heatmap mask. Defaults to 1.0.
        """
        gradcam_misclassified(
            number=number,
            experiment=self,
            target_layer=target_layer,
            dataloader=self.val_loader,
            device="cuda" if torch.cuda.is_available() else "cpu",
            save_path=save_path,
            figsize=figsize,
            class_labels=class_labels,
            mean=mean,
            std=std,
            opacity=opacity,
        )

    def plot_scalars(
        self,
        figsize: Tuple[int, int] = (15, 10),
        save_path: str = None,
    ):
        """
        Plots the tenorboard scalars as a matplotlib figure.

        Args:
            figsize (Tuple[int, int], optional): The size of the figure. Defaults to (15, 10).
            save_path (str, optional): The path to save tht plot to. Defaults to None.
        """

        plot_scalars(self.log_dir, figsize=figsize, save_path=save_path)

    def get_solver(self) -> "BaseSolver":
        """
        Getter method for the solver.

        Returns:
            "BaseSolver": The solver used in this experiment.
        """
        return self.solver

    def get_model(self) -> nn.Module:
        """
        Getter for the model.

        Returns:
            nn.Module: The model.
        """
        return self.solver.model

    @staticmethod
    def builder() -> "ExperimentBuilder":
        """
        Returns an object of the builder interface. Needs to be called if one wants
        to use the builder pattern to define the ``Experiment``.

        Returns:
            ExperimentBuilder: The builder interface for ``Experiment``.
        """
        return ExperimentBuilder()


class ExperimentBuilder:
    def __init__(self, name: str = None, parent: "ExperimentsBuilder" = None):
        """
        A Builder interface for the :class:`Experiment`.

        Args:
            name (str, optional): The name of the experiment. Defaults to None.
            parent (Buildable, optional): The parent :class:`ExperimentsBuilder`, if any. \
                Defaults to None.
        """

        self._props = {}
        self._data = {}
        self._solver = {}
        self.parent = parent

        if name is not None:
            self.name(name)

    def build(self) -> Union[Experiment, "ExperimentsBuilder"]:
        """
        Builds The :class:`Experiment` object.

        Returns:
            Union[Experiment, ExperimentsBuilder]: The built object if there is no parent \
                :class:`ExperimentsBuilder` or the parent.
        """

        # assertion checks
        assert self._solver.get("model") is not None, "Set the model to use."
        assert self._solver.get("optimizer") is not None, "Set the optimizer to use."
        assert self._props.get("name") is not None, "Set the name of the experiment."
        assert (
            self._props.get("epochs") is not None
        ), "Set the number of epochs to train for."
        assert self._data.get("train") is not None, "Set the train dataloader."
        assert self._data.get("val") is not None, "Set the validation dataloader."
        assert (
            self._props.get("log_dir") is not None or self.parent is not None
        ), "Set the log directory for the experiment."

        # building the ``Experiment`` object
        model = self._solver["model"]
        optimizer = self._solver["optimizer"]["cls"](
            model.parameters(),
            *self._solver["optimizer"]["args"],
            **self._solver["optimizer"]["kwargs"],
        )
        scheduler = (
            self._solver["scheduler"]["cls"](
                optimizer,
                *self._solver["scheduler"]["args"],
                **self._solver["scheduler"]["kwargs"],
            )
            if self._solver.get("scheduler") is not None
            else None
        )

        solver = self._solver["cls"](
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            **self._solver["other_args"],
        )

        exp = Experiment(
            name=self._props["name"],
            solver=solver,
            epochs=self._props["epochs"],
            train_loader=self._data["train"],
            val_loader=self._data["val"],
            trainer_args=self._props.get("trainer_args", {}),
            log_dir=self._props.get("log_dir")
            or os.path.join(self.parent.get_log_dir(), self._props["name"]),
            resume_from_checkpoint=self._props.get("resume_from_checkpoint"),
            force_restart=self._props.get("force_restart", False),
        )

        # if there is a parent `ExperimentsBuilder`, sending the created ``Experiment``
        # to be handled by the parent and returning the parent.
        if self.parent is not None:
            self.parent.handle(exp)
            return self.parent

        # else returning the created ``Experiment`` object.
        return exp

    def name(self, name: str) -> "ExperimentBuilder":
        """
        Sets the name of the experiment.

        Args:
            name (str): The name of the experiment

        Returns:
            ExperimentBuilder: Object of this class.
        """

        self._props["name"] = name
        return self

    def log_directory(self, path: str) -> "ExperimentBuilder":
        """
        Sets the log directory of the experiment.

        Args:
            path (str): The path to the log directory of the experiment

        Returns:
            ExperimentBuilder: Object of this class.
        """

        assert (
            self._props.get("name") is not None
        ), "Set the name of the experiment first."

        self._props["log_dir"] = os.path.join(path, self._props["name"])
        return self

    def trainer_args(self, args: Dict[str, Any]) -> "ExperimentBuilder":
        """
        Sets the additional ``pl.Trainer`` args to be used.

        Args:
            args (Dict[str, Any]): The arguments.

        Returns:
            ExperimentBuilder: Object of this class.
        """

        self._props["trainer_args"] = args
        return self

    def train_loader(self, loader: DataLoader) -> "ExperimentBuilder":
        """
        Sets the train dataloader to be used.

        Args:
            loader (DataLoader): The train dataloader.

        Returns:
            ExperimentBuilder: Object of this class.
        """

        self._data["train"] = loader
        return self

    def val_loader(self, loader) -> "ExperimentBuilder":
        """
        Sets the validation dataloader to be used.

        Args:
            loader (DataLoader): The validation dataloader.

        Returns:
            ExperimentBuilder: Object of this class.
        """

        self._data["val"] = loader
        return self

    def solver(self, cls: BaseSolver) -> "ExperimentBuilder":
        """
        The solver to use.

        Args:
            cls (BaseSolver): The class (not object) of the solver to use.

        Returns:
            ExperimentBuilder: Object of this class.
        """

        self._solver["cls"] = cls
        self._add_methods(cls)
        return self

    def epochs(self, epochs: int) -> "ExperimentBuilder":
        """
        Sets the total number of epochs to train for.

        Args:
            epochs (int): The number of epochs.

        Returns:
            ExperimentBuilder: Object of this class.
        """

        self._props["epochs"] = epochs
        return self

    def resume_from_checkpoint(self, path: str) -> "ExperimentBuilder":
        """
        Sets the checkpoint to resume from.

        Args:
            path (str): The path to resume from.

        Returns:
            ExperimentBuilder: Object of this class.
        """

        self._props["resume_from_checkpoint"] = path
        return self

    def force_restart(self, value: bool = True) -> "ExperimentBuilder":
        """
        Deletes old log directory and starts training from scratch.
        """

        self._props["force_restart"] = value
        return self

    def props(self) -> "ExperimentBuilder":
        """
        Defines the properties section of the builder

        Returns:
            ExperimentBuilder: Object of this class.
        """

        return self

    def data(self) -> "ExperimentBuilder":
        """
        Defines the data section (train and validation loaders) of the builder.

        Returns:
            ExperimentBuilder: Object of this class.
        """

        return self

    def _add_methods(self, cls):
        """
        Adds the parameters of the ``cls``'s ``__init__`` method as functions that can
        be used to set the value of that parameter, to ``self``.

        Raises:
            ValueError: When the solver used does not have "model", "optimizer" and \
                "scheduler" as parameters.
        """
        # getting signature of the __init__ function of the class
        sig = signature(cls.__init__)

        # checking if "model", "optimizer", and "scheduler" are arguments
        # in the __init__ function
        if (
            not "model" in sig.parameters
            and "optimizer" in sig.parameters
            and "scheduler" in sig.parameters
        ):
            raise ValueError(
                "To use the builder, the solver's __init__ function should have 'model', 'optimizer', and 'scheduler' as parameters (and in the same order)."
            )

        # defining intializer methods
        def _basic_intializer(name, dest):
            def wrapper(value):
                dest[name] = value
                return self

            return wrapper

        def _optim_initializer(name):
            def wrapper(cls, *args, **kwargs):
                self._solver[name] = {"cls": cls, "args": args, "kwargs": kwargs}
                return self

            return wrapper

        # adding the functions to the object
        self._solver["other_args"] = {}
        for param in sig.parameters.values():
            name = param.name
            if name == "self":
                continue

            if name == "optimizer" or name == "scheduler":
                setattr(self, name, _optim_initializer(name))
            else:
                setattr(
                    self,
                    name,
                    _basic_intializer(
                        name,
                        self._solver if name == "model" else self._solver["other_args"],
                    ),
                )
