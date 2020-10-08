import os
from inspect import signature
from typing import Any, Dict, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from athena.solvers.base_solver import BaseSolver
from athena.utils import ProgbarCallback
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
    ):
        """
        Bundles information regarding an experiment. An experiment is performed on a model, with
        a ``Solver`` (subclass of :class:`BaseSolver`.).

        Args:
            name (str): The name of the experiment.
            model (nn.Module): The model that the experiment is acting on.
            solver (BaseSolver): The solver the experiment is using.
            epochs (int): The number of epochs to train for.
            train_loader (DataLoader): The train ``DataLoader``.
            val_loader (DataLoader): The validation ``DataLoader``.
            log_dir (str): The log directory.
            trainer_args (Dict[str, Any]): Additional arguments to be given to ``pl.Trainer``.
        """

        self.name = name
        self.solver = solver
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_dir = log_dir

        if "gpus" not in trainer_args and torch.cuda.is_available():
            trainer_args["gpus"] = 1

        if "callbacks" in trainer_args:
            trainer_args["callbacks"].append(ProgbarCallback())
        else:
            trainer_args["callbacks"] = [ProgbarCallback()]

        tensorboard_logger = TensorBoardLogger(log_dir, name="")
        self.trainer = pl.Trainer(
            max_epochs=epochs, logger=tensorboard_logger, **trainer_args
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
            **self._solver["optimizer"]["kwargs"]
        )
        scheduler = (
            self._solver["scheduler"]["cls"](
                optimizer,
                *self._solver["scheduler"]["args"],
                **self._solver["scheduler"]["kwargs"]
            )
            if self._solver.get("scheduler") is not None
            else None
        )

        solver = self._solver["cls"](
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            **self._solver["other_args"]
        )

        exp = Experiment(
            name=self._props["name"],
            solver=solver,
            epochs=self._props["epochs"],
            train_loader=self._data["train"],
            val_loader=self._data["val"],
            trainer_args=self._props.get("trainer_args", {}),
            log_dir=self._props.get(
                "log_dir", os.path.join(self.parent.get_log_dir(), self._props["name"])
            ),
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

        self._props["log_dir"] = path
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
