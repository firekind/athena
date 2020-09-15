import abc
from typing import Callable, Type

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


class BaseSolver(abc.ABC):
    def __init__(self, model: nn.Module):
        """
        The base class for a ``Solver``.

        Args:
            model (nn.Module): The model the solver should act on.
        """

        self.model = model
        self.experiment = None
        self._device = "cpu"
        self._optimizer: Optimizer = None
        self._scheduler: LRScheduler = None
        self._train_loader: DataLoader = None
        self._test_loader: DataLoader = None
        self._epochs: int = None
        self._loss_fn: Callable = None
        self._acc_fn: Callable = None
        self._use_tqdm: bool = False

    @abc.abstractmethod
    def train(self, *args):
        """
        Trains the model.
        """
        

    def set_experiment(self, experiment: "Experiment"):
        """
        Sets the experiment that is attached to this solver.

        Args:
            experiment (Experiment): The ``Experiment`` object.
        """
        self.experiment = experiment

    def optimizer(self, optimizer_cls: Type[Optimizer], **kwargs) -> "BaseSolver":
        """
        Sets the optimizer to use. Used in the builder api.

        Args:
            optimizer_cls (Type[Optimizer]): The optimizer class (not the object)
            **kwargs: The keyword arguments to be passed onto the optimizer.

        Raises:
            AssertionError: Raised when there is no model attached to this experiment

        Returns:
            Experiment: object of this class.
        """

        self._optimizer = optimizer_cls(self.model.parameters(), **kwargs)
        return self

    def scheduler(self, scheduler_cls: Type[LRScheduler], **kwargs) -> "BaseSolver":
        """
        Sets the scheduler to use. Used in the builder api.

        Args:
            scheduler_cls (Type[_LRScheduler]): The scheduler class (not the object)
            **kwargs: The keyword arguments to be passed onto the scheduler.

        Raises:
            AssertionError: Raised when there is no optimizer attached to this experiment.

        Returns:
            Experiment: object of this class.
        """

        assert self._optimizer is not None, "Set the optimizer before setting scheduler"

        self._scheduler = scheduler_cls(self._optimizer, **kwargs)
        return self

    def epochs(self, epochs: int) -> "BaseSolver":
        """
        Sets the number of epochs to train for. Used in the builder api.

        Args:
            epochs (int): The epochs to train for.

        Returns:
            Experiment: object of this class.
        """

        self._epochs = epochs
        return self

    def train_loader(self, train_loader: DataLoader) -> "BaseSolver":
        """
        The ``DataLoader`` to use while training the model. Used in the builder api.

        Args:
            train_loader (DataLoader): The dataloader.

        Returns:
            Experiment: object of this class.
        """

        self._train_loader = train_loader
        return self

    def test_loader(self, test_loader: DataLoader) -> "BaseSolver":
        """
        The ``DataLoader`` to use while testing the model. Used in the builder api.

        Args:
            test_loader (DataLoader): The dataloader

        Returns:
            Experiment: object of this class.
        """

        self._test_loader = test_loader
        return self

    def use_tqdm(self) -> "BaseSolver":
        """
        Uses tqdm for the progress bar, instead of the keras style progress bar. \
            Used in the builder api.

        Returns:
            Experiment: object of this class.
        """

        self._use_tqdm = True
        return self

    def loss_fn(self, loss_fn: Callable) -> "BaseSolver":
        """
        The loss function to use. Used in the builder api.

        Args:
            loss_fn (Callable): The loss function

        Returns:
            Experiment: object of this class
        """

        self._loss_fn = loss_fn
        return self

    def acc_fn(self, acc_fn: Callable) -> "BaseSolver":
        """
        The accuracy function to use. Used in the builder api.

        Args:
            acc_fn (Callable): The accuracy function

        Returns:
            Experiment: object of this class
        """

        self._acc_fn = acc_fn
        return self

    def device(self, device: str) -> "BaseSolver":
        """
        The device to use. Used in the builder api.

        Args:
            device (str): A valid pytorch device string.

        Raises:
            AssertionError: Raised when there is no model attached to this experiment.

        Returns:
            Experiment: object of this class.
        """

        self._device = device
        self.model.to(device)
        return self

    def build(self) -> "Experiment":
        """
        Completes the build phase of the solver. Used in the builder api.

        Returns:
            Experiment: The experiment attached to this solver.
        """

        return self.experiment