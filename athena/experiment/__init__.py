import os
from pathlib import Path
from typing import Dict

import torch.nn as nn
from athena.builder import Buildable
from athena.solvers.base_solver import BaseSolver


class Experiment:
    def __init__(self, name: str, model: nn.Module, solver: BaseSolver):
        """
        Bundles information regarding an experiment. An experiment is performed on a model, with
        a ``Solver``.

        Args:
            name (str): The name of the experiment.
            model (nn.Module): The model that the experiment is acting on.
            solver (BaseSolver): The solver the experiment is using.
        """

        self.name = name
        self.model = model
        self.solver = solver

    def run(self):
        """
        Runs the experiment. More specifically, calls the ``BaseSolver.train`` method and saves the
        ``History``.
        """

        print("\033[1m\033[92m=> Running experiment: %s\033[0m" % self.name, flush=True)

        self.history = self.solver.train()

    def get_solver(self) -> BaseSolver:
        """
        Getter method for the solver.

        Returns:
            BaseSolver: The solver used in this experiment.
        """
        return self.solver

    def get_model(self) -> nn.Module:
        """
        Getter for the model.

        Returns:
            nn.Module: The model.
        """
        return self.model

    @staticmethod
    def builder(parent: Buildable = None) -> "ExperimentBuilder":
        """
        Returns an object of the builder interface. Needs to be called if one wants
        to use the builder pattern to define the ``Experiment``.

        Args:
            parent (Buildable, optional): The parent builder interface. Defaults to None.

        Returns:
            ExperimentBuilder: The builder interface for ``Experiment``.
        """
        return ExperimentBuilder(parent)


class ExperimentBuilder(Buildable):
    def __init__(self, parent: Buildable, name: str = None):
        """
        A Builder interface for the :class:`Experiment`.

        Args:
            parent (Buildable, optional): The parent builder interface.
            name (str, optional): The name of the experiment.
        """
        super().__init__(parent)

        self._solver: BaseSolver = None
        self._model = None
        self._name = name

        # adding the correct log directory to context if present
        # in parent
        parent_log_dir = self.find_in_context("log_dir")
        if parent_log_dir is not None and name is not None:
            self.add_to_context("log_dir", os.path.join(parent_log_dir, self._name))

    def create(self) -> Experiment:
        """
        Creates and returns an object of :class:`Experiment`.

        Returns:
            Experiment
        """
        # asserting the values
        assert self._name is not None, "Set the name of the experiment."
        assert (
            self._model is not None or self.find_in_context("model") is not None
        ), "Set the model for the experiment."
        assert (
            self.find_in_context("log_dir") is not None
        ), "Set the log directory for the experiment."

        # creating the experiment object
        return Experiment(
            name=self._name,
            model=self._model or self.find_in_context("model"),
            solver=self._solver,
        )

    def handle(self, obj: object):
        """
        Handles objects returned by the :func:`create` method of child interfaces.
        Used to recieve the :class:`BaseSolver` object from the Solver child interface.

        Args:
            obj (object): The object returned by the child interface.
        """
        if isinstance(obj, BaseSolver):
            self._solver = obj

    def log_directory(self, path: str) -> "ExperimentBuilder":
        """
        Sets the log directory of the experiment.

        Args:
            path (str): The path to the directory.

        Returns:
            ExperimentBuilder: Object of this class.
        """
        assert self._name is not None, "Set the name of the experiment first."

        self.add_to_context("log_dir", os.path.join(path, self._name))
        return self

    def name(self, name: str) -> "ExperimentBuilder":
        """
        Sets the name of the experiment

        Returns:
            str: The name of the experiment.
        """

        self._name = name
        return self

    def model(self, model: nn.Module) -> "ExperimentBuilder":
        """
        Sets the model to use.

        Returns:
            nn.Module: The model.
        """

        self._model = model
        self.add_to_context("model", model)

        return self

    def solver(self, solver_cls: BaseSolver) -> Buildable:
        """
        Returns the builder interface of the given solver class.

        Args:
            solver_cls (BaseSolver): The solver class

        Returns:
            Buildable: The builder interface.
        """

        return solver_cls.builder(self)


class Experiments:
    def __init__(self, name: str, experiments: Dict, log_dir: str = None):
        """
        Defines a list of experiments that has to be run one after the other.

        Args:
            name (str): The name of the list of experiments.
            experiments (Dict): The name-experiment key value pairs.
            log_dir (str, optional): The parent of the directory where the logs should be stored. \
                the directory where the logs will be stored will be the ``name`` parameter.
        """

        self.name = name
        self.experiments = experiments
        self.log_dir = log_dir

    def run(self):
        """
        Runs all the experiments.
        """

        for _, exp in self.experiments.items():
            exp.run()

    def __getitem__(self, name):
        return self.experiments[name]

    def __iter__(self):
        for name, obj in self.experiments.items():
            yield obj

    @staticmethod
    def builder(parent: Buildable = None):
        """
        Returns an object of the builder interface. Needs to be called if one wants
        to use the builder pattern to define the ``Experiments``.

        Args:
            parent (Buildable, optional): The parent builder interface. Defaults to None.

        Returns:
            ExperimentsBuilder: The builder interface for ``Experiments``.
        """
        return ExperimentsBuilder(parent)


class ExperimentsBuilder(Buildable):
    def __init__(self, parent: Buildable = None):
        """
        A Builder interface for the :class:`Experiments`.

        Args:
            parent (Buildable, optional): The parent builder interface.
        """
        super().__init__(parent, ["name"])

        self.experiments = {}
        self._log_directory = None

    def add(self, name) -> ExperimentBuilder:
        """
        Adds an experiment to the list.

        Args:
            name (str): The name of the experiment.

        Returns:
            ExperimentBuilder: The builder interface for :class:`Experiment`
        """

        return ExperimentBuilder(self, name)

    def create(self) -> Experiments:
        """
        Creates and returns an object of :class:`Experiments`.

        Returns:
            Experiments
        """

        # assertion checks
        assert self.get_name() is not None, "Set the name of this group of experiments."
        assert self._log_directory is not None, "Set the log directory."

        # creating `Experiments` object.
        return Experiments(self.get_name(), self.experiments, self._log_directory)

    def handle(self, obj: object):
        """
        Handles objects returned by the :func:`create` method of child interfaces.
        Used to recieve the :class:`Experiment` object from the ExperimentBuilder
        child interface.

        Args:
            obj (object): The object returned by the child interface.
        """
        if isinstance(obj, Experiment):
            self.experiments[obj.name] = obj

    def log_directory(self, path) -> "ExperimentsBuilder":
        """
        Sets the log directory of the experiment.

        Args:
            path (str): The path to the directory.

        Returns:
            ExperimentsBuilder: Object of this class.
        """
        assert (
            self.get_name() is not None
        ), "Set the name of this group of experiments before setting log directory."

        self._log_directory = path
        self.add_to_context("log_dir", os.path.join(path, self.get_name()))

        return self

    def model(self, model: nn.Module) -> "ExperimentsBuilder":
        """
        Sets the model for all the experiments. Can be overriden per experiment.

        Args:
            model (nn.Module): The model.

        Returns:
            ExperimentsBuilder: Object of this class
        """

        self.add_to_context("model", model)
        return self
