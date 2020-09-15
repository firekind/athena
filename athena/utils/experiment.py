from typing import Dict, Any, Type, Callable, Union

import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from . import History
from athena.solvers.base_solver import BaseSolver


class Experiment:
    def __init__(
        self,
        name: str,
        model: nn.Module = None,
        solver_cls: Type[BaseSolver] = None,
        train_args: Dict[str, Any] = None,
    ):
        """
        Bundles information regarding an experiment. An experiment is performed on a model, with
        a ``Solver`` and the arguments that have to be used to train the model.

        Args:
            name (str): The name of the experiment.
            model (nn.Module, optional): The model to experiment on. Defaults to None.
            solver_cls (BaseSolver, optional): The ``Solver`` class to use. Defaults to None.
            train_args (Dict[str, Any], optional): The arguments to be passed on to the ``Solver`` when training. \
                Defaults to None.
        """

        # the name of the experiment
        self.name = name

        # the model object
        self.net = model

        # creating the object of the solver if the class is not None
        if solver_cls is not None:
            self.solver_obj = solver_cls(model)
        else:
            self.solver_obj = None

        # the arguments to be passed to the solver while training. If it is None,
        # initialize it to an empty dictionary
        self.train_args = train_args
        if self.train_args is None:
            self.train_args = {}

        # the history object, stores info about the losses and accuracies
        # once the training is over.
        self.history: History = None

        # the chain is used when ``Experiments`` class is used to define multiple experiments
        # to run. This chain attribute keeps track of the ``Experiments`` object that was
        # used to define the experiments. If the ``Experiment`` was defined without using
        # ``Experiments``, this attribute will remain None.
        # The ``build`` function returns this attribute if it is not none, so that more
        # experiments can be added to the ``Experiments`` object.
        self._chain = None

    def run(self):
        """
        Runs the experiment. More specifically, calls the ``BaseSolver.train`` method and saves the
        ``History``.
        """

        print(
            "\033[1m\033[92m=> Running experiment: %s\033[0m" % self.name, flush=True
        )

        self.history = self.solver_obj.train(**self.train_args)

    def model(self, model: nn.Module) -> "Experiment":
        """
        Sets the model to use.

        Args:
            model (nn.Module): The model to use.

        Returns:
            Experiment: object of this class.
        """
        self.net = model
        return self

    def solver(self, solver: Type[BaseSolver]) -> "Experiment":
        """
        Sets the solver to use.

        Args:
            solver (Type[BaseSolver]): The solver class (not object)

        Raises:
            AssertionError: When the model for the experiment is not defined.

        Returns:
            Experiment: object of this class.
        """
        assert self.net is not None, "Set the model before setting the solver."

        self.solver_obj = solver(self.net)
        self.solver_obj.set_experiment(self)
        return self.solver_obj

    def build(self) -> Union["Experiment", "Experiments"]:
        """
        Prepares the experiment, so that it can be run.

        Returns:
            Union[Experiment, Experiments]: The object of this class, if an ``Experiments`` object was \
                not used to create this experiment, else the ``Experiments`` object used to create this \
                experiment.
        """

        assert (
            self.model is not None and self.solver_obj is not None
        ), "Model and solver should be specified"

        # returning self, since this experiment was defined as a standalone experiment
        if self._chain is None:
            return self

        # else returning the chain this experiment is attached to
        else:
            return self._chain

    def _set_chain(self, chain: "Experiments"):
        """
        Sets the chain that this experiment should attach itself to.

        Args:
            chain (Experiments): The chain object.
        """

        self._chain = chain


class Experiments:
    def __init__(self):
        """
        Defines a list of experiments that has to be run one after the other.
        """

        self.experiments = {}

    def add(self, name: str) -> Experiment:
        """
        Adds an experiment to the chain of experiments.

        Args:
            name (str): The name of the experiment.

        Returns:
            Experiment: The ``Experiment`` object that was added to the chain.
        """

        e = Experiment(name)
        self._add_experiment(e)

        return e

    def run(self):
        """
        Runs all the experiments.
        """

        for _, exp in self.experiments.items():
            exp.run()

    def _add_experiment(self, experiment: "Experiment"):
        """
        Adds the experiment to the chain.

        Args:
            experiment (Experiment): The experiment to add.
        """

        # sets the experiment's chain to the object of this class.
        experiment._set_chain(self)

        # adding the experiment to the list of experiments
        self.experiments[experiment.name] = experiment

    def done(self) -> "Experiments":
        """
        Signifies the end of the chain of experiments.

        Returns:
            Experiments: Object of this class.
        """

        return self

    def __getitem__(self, name):
        return self.experiments[name]

    def __iter__(self):
        for name, obj in self.experiments.items():
            yield obj