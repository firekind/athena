import os
import abc
from pathlib import Path
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
from athena.utils import Checkpoint, Checkpointable
from .history import History


class _BaseSolverMeta(abc.ABCMeta):
    """
    Metaclass for the :class:`BaseSolver`.
    """

    def __call__(cls, *args, **kwargs):
        # creating the solver object
        obj = type.__call__(cls, *args, **kwargs)

        # initializing the checkpoints after object creation
        obj._init_checkpoint()

        # returning the created object.
        return obj


class BaseSolver(metaclass=_BaseSolverMeta):
    def __init__(
        self,
        model: nn.Module,
        log_dir: str,
        epochs: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        loss_fn: Callable,
        acc_fn: Callable,
        device: str,
        use_tqdm: bool,
        max_to_keep: Union[int, str],
    ):
        """
        The base class of a Solver.

        Args:
            model (nn.Module): The model to train.
            log_dir (str): The directory to store the logs.
            epochs (int): The number of epochs to train for.
            train_loader (DataLoader): The ``DataLoader`` for the training data.
            test_loader (DataLoader): The ``DataLoader`` for the test data.
            optimizer (Optimizer): The optimizer to use.
            scheduler (LRScheduler, optional): The scheduler to use. Defaults to None.
            loss_fn (Callable, optional): The loss function to use. If ``None``, the :meth:`default_loss_fn` \
                will be used. Defaults to None.
            device (str, optional): A valid pytorch device string. Defaults to "cpu".
            use_tqdm (bool, optional): Whether to use tqdm progress bar. Defaults to False.
            max_to_keep (Union[int, str], optional): The max number of checkpoints to keep. Defaults to "all".
        """

        self.model = model
        self.log_dir = log_dir
        self.checkpoint_dir = os.path.join(log_dir, "checkpoints")
        self.tensorboard_dir = os.path.join(log_dir, "tensorboard")
        self.epochs = _CheckpointableEpoch(epochs, None)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.acc_fn = acc_fn
        self.device = device
        self.use_tqdm = use_tqdm
        self.max_to_keep = max_to_keep

        self.experiment = None
        self.progbar = None
        self.history = History(epochs)
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
        self.checkpoint = None

        # creating log directory
        self._create_log_directory()

    @abc.abstractmethod
    def train(self):
        """
        Trains the model.
        """

        # restoring latest checkpoint
        self.checkpoint.restore()

        # checking to see if experiment has already been completed
        if self.epochs.get_current_epoch() >= self.epochs:
            print("Experiment has already been completed.\n", flush=True)
            return

        # adding model to graph if training is happening for the first time
        if self.epochs.get_current_epoch() == -1:
            self.model.eval()

            if hasattr(self.train_loader.dataset, "input_shape"):
                input_shape = (1,) + self.train_loader.dataset.input_shape
            else:
                images, labels = next(iter(self.train_loader))
                input_shape = images.shape

            self.writer.add_graph(
                self.model,
                torch.randn(
                    input_shape, device=self.device
                ),  # using a random tensor as input since using the image from the dataset sometimes causes jit trace warnings
            )

            self.model.train()

        # setting default loss function in case loss function is not specified
        if self.loss_fn is None:
            self.loss_fn = self.default_loss_fn()

    @abc.abstractmethod
    def default_loss_fn(self) -> Callable:
        """
        The default loss function to use, in case no loss function is set.

        Returns:
            Callable: The default loss function to use
        """

    @abc.abstractmethod
    def track(self) -> List:
        """
        The objects to checkpoint, apart from the model, optimizer, scheduler and epoch.

        Returns:
            List: The list of objects to checkpoint.
        """
        pass

    def _init_checkpoint(self):
        """
        Initializes and creates the checkpoint object.
        """

        # getting the list of objects to checkpoint
        objs = self.track() + [
            self.history,
            self.epochs,
            self.model,
            self.optimizer,
        ]

        # adding the scheduler to the list if it is not None
        if self.scheduler is not None:
            objs += [self.scheduler]

        # creating the checkpoint.
        self.checkpoint = Checkpoint(
            self.checkpoint_dir,
            objs,
            self.max_to_keep,
        )

        self.epochs.set_checkpoint(self.checkpoint)

    def _create_log_directory(self):
        """
        Creates the log directory and its parent.
        """

        # creating the parent directory
        Path(self.log_dir).mkdir(exist_ok=True, parents=True)

    def get_history(self) -> "History":
        """
        Getter for the history object of the solver.

        Returns:
            History: The history object.
        """

        return self.history

    def cleanup(self):
        """
        Performs any cleanup.
        """

        self.writer.close()

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
                self.history.add_metric(label, value, self.epochs.get_current_epoch())
                self.writer.add_scalar(
                    label, value, self.epochs.get_current_epoch()
                )

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
                if self.use_tqdm:
                    self.progbar = tqdm(
                        self.train_loader if train else self.test_loader
                    )
                else:
                    self.progbar = Kbar(
                        len(self.train_loader) if train else len(self.test_loader)
                    )

                # executing function
                res: StepResult = func(self, *args, **kwargs)

                # closing progress bars
                if not self.use_tqdm:
                    self.progbar.add(1, values=res.data)
                else:
                    self.progbar.close()

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
            if self.use_tqdm:
                desc = " - ".join([f"{name}: {value:0.4f}" for name, value in res.data])
                self.progbar.set_description(
                    desc=f"Batch_id: {res.batch_idx + 1} - {desc}"
                )
                self.progbar.update(1)
            else:
                self.progbar.update(res.batch_idx, values=res.data)

            # returning the result of the function
            return res

        return wrapper

    @staticmethod
    def train_step(func: Callable) -> Callable:
        """
        A convenience decorator that is a combination of \
            :func:`log_results` decorator and :func:`prog_bar` decorator.

        Args:
            func (Callable): The function to decorate.

        Returns:
            Callable: The decorated function.
        """

        @wraps(func)
        @BaseSolver.log_results
        @BaseSolver.prog_bar()
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        return wrapper


class StepResult:
    def __init__(self, data: List[Tuple[str, Any]], **kwargs):
        """
        Results of a train step (train epoch)

        Args:
            data (List[Tuple[str, Any]]): The results. Typically a list of \
                label-accuracy/loss pairs
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
            data (List[Tuple[str, Any]]): The results of training on the batch. Typically \
                a list of label-accuracy/loss pairs.
        """

        self.batch_idx = batch_idx
        self.data = data
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __getitem__(self, key):
        return getattr(self, key)


class _CheckpointableEpoch(Checkpoint):
    def __init__(self, epochs: int, checkpoint: Checkpoint):
        """
        An epoch value that is checkpointable.

        Args:
            epochs (int): The total number of epochs.
            checkpoint (Checkpoint): The checkpoint object whose :meth:`save` is called\
                after every epoch.
        """

        self.epochs = epochs
        self.checkpoint = checkpoint
        self.current = -1
        self.just_restored = False

    def state_dict(self):
        return {"epoch": self.current}

    def load_state_dict(self, data):
        self.current = data["epoch"]
        self.just_restored = True
    
    def get_current_epoch(self):
        return self.current

    def set_checkpoint(self, value):
        self.checkpoint = value

    def __iter__(self):
        return self

    def __next__(self):
        # save checkpoint if training was not just started
        # or the checkpoint was not just restored
        if self.current != -1 and not self.just_restored:
            self.checkpoint.save()
        
        # if the checkpoint was just restored, clear the 
        # flag
        if self.just_restored:
            self.just_restored = False

        self.current += 1

        if self.current < self.epochs:
            return self.current
        
        raise StopIteration

    def __repr__(self):
        return str(self.epochs)

    def __int__(self):
        return self.epochs

    def __le__(self, value):
        return self.epochs <= value

    def __lt__(self, value):
        return self.epochs < value

    def __ge__(self, value):
        return self.epochs >= value

    def __gt__(self, value):
        return self.epochs > value

    def __eq__(self, value):
        return self.epochs == value

    def __ne__(self, value):
        return self.epochs != value
