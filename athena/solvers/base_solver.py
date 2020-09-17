import abc
from functools import wraps
from typing import Any, Callable, List, Tuple, Type, Union

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from athena.utils.progbar import Kbar


def _writer_wrapper(func: Callable) -> Callable:
    """
    Used to wrap ``SummaryWriter`` related functions. The decorated function won't \
        execute if the writer object of the solver is None.
    Args:
        func (Callable): The function to decorate.

    Returns:
        Callable: The decorated function.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.writer is not None:
            func(self, *args, **kwargs)

    return wrapper


class BaseSolver(abc.ABC):
    def __init__(
        self,
        model: nn.Module,
        epochs: int = None,
        train_loader: DataLoader = None,
        test_loader: DataLoader = None,
        optimizer: Optimizer = None,
        scheduler: LRScheduler = None,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        acc_fn: Callable = None,
        device: str = None,
        use_tqdm: bool = False,
        log_dir: str = None,
    ):
        """
        The base class for a ``Solver``.

        Args:
            model (nn.Module): The model the solver should act on.
            epochs (int, optional): The number of epochs to train for. Defaults to None.
            train_loader (DataLoader, optional): The ``DataLoader`` for the training data. Defaults to None.
            test_loader (DataLoader, optional): The ``DataLoader`` for the test data. Defaults to None.
            optimizer (Optimizer, optional): The optimizer to use. Defaults to None.
            scheduler (LRScheduler, optional): The ``LRScheduler`` to use. Defaults to None.
            loss_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional): The loss function to use. Defaults to None.
            acc_fn (Callable, optional): The accuracy function to use. Defaults to None.
            device (str, optional): A valid pytorch device string. Defaults to None.
            use_tqdm (bool, optional): If True, uses tqdm instead of a keras style progress bar (``pkbar``). Defaults to False.
            log_dir (str, optional): The directory to store the logs. Defaults to None.
        """

        self.model = model
        self._epochs: int = None
        self._train_loader: DataLoader = None
        self._test_loader: DataLoader = None
        self._optimizer: Optimizer = None
        self._scheduler: LRScheduler = None
        self._loss_fn = loss_fn
        self._acc_fn = acc_fn
        self._device = "cpu"
        self._use_tqdm: bool = False
        self.log_dir = log_dir

        self.experiment = None
        self.history = None
        self.current_epoch = 0
        self.writer: SummaryWriter = (
            None if log_dir is None else SummaryWriter(log_dir=log_dir)
        )
        self.progbar = None

    @abc.abstractmethod
    def train(self, *args):
        """
        Trains the model.
        """

    @_writer_wrapper
    def writer_add_model(self, model: nn.Module, input_data: torch.Tensor):
        """
        Adds the model to tensorboard

        Args:
            model (nn.Module): The model to add
            input_data (torch.Tensor): The image data to be used to forward prop.
        """

        self.writer.add_graph(model, input_data)
        self.writer.flush()

    @_writer_wrapper
    def writer_add_scalar(self, tag: str, value: float, step: int):
        """
        Adds a scalar to tensorboard.

        Args:
            tag (str): The tag of the scalar.
            value (float): The value.
            step (int): The step count at which the scalar should be added.
        """
        self.writer.add_scalar(tag, value, step)

    @_writer_wrapper
    def writer_close(self):
        """
        Closes tensorboard writer.
        """

        self.writer.close()

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

    def get_model(self) -> nn.Module:
        """
        Getter for the model.

        Returns:
            nn.Module: The model
        """

        return self.model

    def get_epochs(self) -> int:
        """
        Getter for the number of epochs.

        Returns:
            int: The number of epochs to train for.
        """

        return self._epochs

    def get_train_loader(self) -> DataLoader:
        """
        Getter for the train data loader.

        Returns:
            DataLoader: The data loader for the train dataset.
        """

        return self._train_loader

    def get_test_loader(self) -> DataLoader:
        """
        Getter for the test data loader

        Returns:
            DataLoader: The data loader for the test dataset.
        """

        return self._test_loader

    def get_optimizer(self) -> Optimizer:
        """
        Getter for the optimizer.

        Returns:
            Optimizer: The optimizer.
        """

        return self._optimizer

    def get_scheduler(self) -> LRScheduler:
        """
        Getter for the scheduler.

        Returns:
            LRScheduler: The scheduler.
        """

        return self._scheduler

    def get_loss_fn(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Getter for the loss function.

        Returns:
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: The loss function to use.
        """

        return self._loss_fn

    def set_loss_fn(self, func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        """
        Setter for the loss function.

        Args:
            func (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The new loss function.
        """

        self._loss_fn = func

    def get_acc_fn(self) -> Callable:
        """
        Getter for the accuracy function.

        Returns:
            Callable: The accuracy function to use.
        """

        return self._acc_fn

    def get_device(self) -> str:
        """
        Getter for the device string.

        Returns:
            str: The device string.
        """

        return self._device

    def should_use_tqdm(self) -> bool:
        """
        Getter for whether tqdm should be used or not.

        Returns:
            bool: Whether tqdm should be used or not.
        """

        return self._use_tqdm

    def get_progbar(self) -> Union[Kbar, tqdm]:
        """
        Getter for the progress bar

        Returns:
            Union[Kbar, tqdm]: The progress bar.
        """

        return self.progbar

    def set_progbar(self, value: Union[Kbar, tqdm]):
        """
        Setter for the progress bar.

        Args:
            value (Union[Kbar, tqdm]): The new progress bar.

        """

        self.progbar = value

    def get_history(self) -> "History":
        """
        Getter for the history object of the solver.

        Returns:
            History: The history object.
        """

        return self.history

    @staticmethod
    def log_results(func: Callable) -> Callable:
        """
        Decorator that is used log the returned value of ``func`` to the
        solver's ``history`` and ``SummaryWriter``.

        Args:
            func (Callable): The decorated function. The return type of the function \
                should be :class:`StepResult`

        Returns:
            Callable: The wrapped function.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # executing the function
            res: StepResult = func(self, *args, **kwargs)

            # storing the returned data into history and
            # summary writer.
            for label, value in res.data:
                self.get_history().add_metric(label, value)
                self.writer_add_scalar(label, value, self.current_epoch)

            # returing the result of the function
            return res

        return wrapper

    @staticmethod
    def prog_bar(train: bool = True) -> Callable:
        """
        Returns a decorator that is used to manage a progress bar.
        The decorator creates the progress bar, executes the function, and closes
        the progress bar.

        The function that is decorated is expected to call a function that is decorated
        with the :meth:`prog_bar_update` decorator.

        Args:
            train (bool, optional): Whether to use the train dataset of the solver or the test dataset. Defaults to True.

        Returns:
            Callable: The decorator.
        """

        def _prog_bar_wrapper(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                # creating the progress bar
                if self.should_use_tqdm():
                    self.set_progbar(
                        tqdm(self._train_loader if train else self._test_loader)
                    )
                else:
                    self.set_progbar(
                        Kbar(
                            len(self._train_loader) if train else len(self._test_loader)
                        )
                    )

                # executing function
                res: StepResult = func(self, *args, **kwargs)

                # closing progress bars
                if not self.should_use_tqdm():
                    self.get_progbar().add(1, values=res.data)
                else:
                    self.get_progbar().close()

                # returing the result of the function
                return res

            return wrapper

        return _prog_bar_wrapper

    @staticmethod
    def prog_bar_update(func: Callable) -> Callable:
        """
        Decorator that updates the progress bar with the data returned by ``func``. This decorator
        will only work if ``func``'s caller is decorated with :meth:`prog_bar` decorator. ``func``
        should return an instance of :class:`BatchResult`.

        Args:
            func (Callable): The function to decorate.

        Returns:
            Callable: The decorated function.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # executing function
            res: StepResult = func(self, *args, **kwargs)

            # updating progress bar
            if self.should_use_tqdm():
                desc = " - ".join([f"{name}: {value:0.4f}" for name, value in res.data])
                self.get_progbar().set_description(
                    desc=f"Batch_id: {res.batch_idx + 1} - {desc}"
                )
                self.get_progbar().update(1)
            else:
                self.get_progbar().update(res.batch_idx, values=res.data)

            # returning the result of the function
            return res

        return wrapper


class StepResult:
    def __init__(self, data: List[Tuple[str, Any]], **kwargs):
        """
        Results of a train step (train epoch)

        Args:
            data (List[Tuple[str, Any]]): The results.
        """

        self.data = data
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __getitem__(self, key):
        return getattr(self, key)


class BatchResult:
    def __init__(self, batch_idx: int, data: List[Tuple[str, Any]], **kwargs):
        """
        Results of training on a batch.

        Args:
            batch_idx (int): The batch index.
            data (List[Tuple[str, Any]]): The results of training on the batch.
        """

        self.batch_idx = batch_idx
        self.data = data
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __getitem__(self, key):
        return getattr(self, key)
