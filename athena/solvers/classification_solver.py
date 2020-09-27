from typing import Callable, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from athena.builder import Buildable
from torch.optim.lr_scheduler import OneCycleLR, Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader

from .base_solver import BaseSolver, BatchResult, StepResult


class ClassificationSolver(BaseSolver):
    def __init__(
        self,
        model: nn.Module,
        log_dir: str,
        epochs: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler = None,
        loss_fn: Callable = None,
        device: str = "cpu",
        use_tqdm: bool = False,
        max_to_keep: Union[int, str] = "all",
    ):
        """
        A solver for classification problems.

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

        super(ClassificationSolver, self).__init__(
            model=model,
            log_dir=log_dir,
            epochs=epochs,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            acc_fn=None,
            device=device,
            use_tqdm=use_tqdm,
            max_to_keep=max_to_keep,
        )

    def train(self):
        """
        Trains the model.

        Args:

        """
        super(ClassificationSolver, self).train()

        # training
        for epoch in range(self.get_start_epoch(), self.epochs):
            print(
                "Epoch: %d / %d" % (epoch + 1, self.epochs),
                flush=self.use_tqdm,
            )

            # performing train step
            self.train_step()

            # stepping scheduler
            if self.scheduler is not None and not isinstance(
                self.scheduler, OneCycleLR
            ):
                self.scheduler.step()

            # performing test step
            if self.test_loader is not None:
                self.test_step()

        self.cleanup()

    @BaseSolver.train_step
    def train_step(self) -> StepResult:
        """
        Performs a single train step.

        Returns:
            StepResult: A :class:`StepResult` object that contains the epoch data.
        """

        # setting model in train mode
        self.model.train()

        # defining variables
        correct = 0
        processed = 0
        train_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            res = self.train_on_batch(
                batch=data.to(self.device),
                target=target.to(self.device),
                batch_idx=batch_idx,
                running_train_loss=train_loss,
                running_correct=correct,
                running_processed=processed,
            )
            correct = res.running_correct
            processed = res.running_processed
            train_loss = res.running_train_loss

        return StepResult(
            data=[
                ("train loss", float(train_loss / len(self.train_loader))),
                (
                    "train accuracy",
                    float(100 * correct / len(self.train_loader.dataset)),
                ),
            ]
        )

    @BaseSolver.prog_bar_update
    def train_on_batch(
        self,
        batch: torch.Tensor,
        target: torch.Tensor,
        batch_idx: int,
        running_train_loss: torch.Tensor,
        running_correct: int,
        running_processed: int,
    ) -> BatchResult:
        """
        Trains the model on a batch.

        Args:
            batch (torch.Tensor): The batch to train on.
            target (torch.Tensor): The targets of the batch.
            batch_idx (int): The batch index
            running_train_loss (torch.Tensor): The running train loss.
            running_correct (int): The running count of correctly classified images.
            running_processed (int): The running count of processed images.

        Returns:
            BatchResult: The results of training.
        """

        # zeroing out accumulated gradients
        self.optimizer.zero_grad()

        # forward prop
        y_pred = self.model(batch)

        # calculating loss
        loss = self.loss_fn(y_pred, target)
        running_train_loss += loss.detach()

        # backpropagation
        loss.backward()
        self.optimizer.step()

        # calculating accuracy
        pred = y_pred.argmax(dim=1, keepdim=True)
        running_correct += pred.eq(target.view_as(pred)).sum().item()
        running_processed += len(batch)
        acc = 100 * running_correct / running_processed

        if isinstance(self.scheduler, OneCycleLR):
            self.scheduler.step()

        return BatchResult(
            batch_idx=batch_idx,
            data=[("train loss", loss.item()), ("train accuracy", acc)],
            running_correct=running_correct,
            running_processed=running_processed,
            running_train_loss=running_train_loss,
        )

    @BaseSolver.log_results
    def test_step(self) -> StepResult:
        """
        Performs a single test step.

        Returns:
            StepResult: The results of the step.
        """

        # setting model to evaluation mode
        self.model.eval()

        # defining variables
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                # casting data to device
                data, target = data.to(self.device), target.to(self.device)

                # forward prop
                output = self.model(data)

                # calculating loss
                test_loss += self.loss_fn(output, target)

                # calculating number of correctly predicted classes
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        # averaging loss
        test_loss /= len(self.test_loader)
        test_acc = 100.0 * correct / len(self.test_loader.dataset)

        # printing result
        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss,
                correct,
                len(self.test_loader.dataset),
                test_acc,
            ),
            flush=self.use_tqdm,
        )

        return StepResult(data=[("test loss", test_loss), ("test accuracy", test_acc)])

    def default_loss_fn(self) -> Callable:
        """
        In case no loss function is specified, setting it to nll loss.

        Returns:
            Callable: The loss function.
        """

        print(
            "\033[1m\033[93mWarning:\033[0m Loss function not specified. Using nll loss.",
            flush=self.use_tqdm,
        )

        return F.nll_loss

    def track(self) -> List:
        """
        List of objects to checkpoint, apart from the model, scheduler, optimizer and epoch.

        Returns:
            List
        """

        return []

    def get_misclassified(
        self, data_loader: DataLoader, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gets the information of the misclassified data items in the given dataset.

        Args:
            test_loader (DataLoader): The ``DataLoader`` to use.
            device (str): A valid pytorch device string.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple consisting of the \
                image information, predicted, and actual class of the misclassified images.
        """

        # defining variables
        misclassified = []
        misclassified_pred = []
        misclassified_target = []

        # put the model to evaluation mode
        self.get_model().eval()

        with torch.no_grad():
            for data, target in data_loader:
                # casting data to device
                data, target = data.to(device), target.to(device)

                # forward prop
                output = self.model(data)

                # get the predicted class
                pred = output.argmax(dim=1, keepdim=True)

                # get the current misclassified in this batch
                list_misclassified = pred.eq(target.view_as(pred)) == False
                batch_misclassified = data[list_misclassified.squeeze()]
                batch_mis_pred = pred[list_misclassified]
                batch_mis_target = target.view_as(pred)[list_misclassified]

                # add data to function variables
                misclassified.append(batch_misclassified)
                misclassified_pred.append(batch_mis_pred)
                misclassified_target.append(batch_mis_target)

        # group all the batched together
        misclassified = torch.cat(misclassified)
        misclassified_pred = torch.cat(misclassified_pred)
        misclassified_target = torch.cat(misclassified_target)

        return misclassified, misclassified_pred, misclassified_target

    @staticmethod
    def builder(parent: Buildable = None):
        """
        Returns an object of the builder interface. Needs to be called if one wants
        to use the builder pattern to define the solver.

        Args:
            parent (Buildable, optional): The parent builder interface. Defaults to None.

        Returns:
            ClassificationSolverBuilder: The builder interface for ``ClassificationSolver``.
        """
        return ClassificationSolverBuilder(parent)


class ClassificationSolverBuilder(Buildable):
    def __init__(self, parent: Buildable = None):
        """
        A Builder interface for the :class:`ClassificationSolver`.

        Args:
            parent (Buildable, optional): The parent builder interface. Defaults to None.
        """

        super(ClassificationSolverBuilder, self).__init__(
            parent,
            args=[
                "model",
                "log_dir",
                "epochs",
                "train_loader",
                "test_loader",
                "optimizer",
                "scheduler",
                "loss_fn",
                "device",
                "use_tqdm",
                "max_checkpoints_to_keep",
            ],
        )

    def create(self) -> ClassificationSolver:
        """
        Creates and returns an object of :class:`ClassificationSolver`.

        Returns:
            ClassificationSolver
        """

        # asserting values
        assert (
            self.get_model() is not None or self.find_in_context("model") is not None
        ), "Set a model to use."
        assert self.get_optimizer() is not None, "Set the optimizer class to use."
        assert self.get_epochs() is not None, "Set the number of epochs to train for."
        assert self.get_train_loader() is not None, "Set the dataloader to train on."
        assert (
            self.get_log_dir() is not None
            or self.find_in_context("log_dir") is not None
        ), "Set the directory to store the logs."

        # getting the model that was defined using this interface or any other
        # parent interface.
        model = self.get_model() or self.find_in_context("model")

        # getting the log directory that was defined using this interface or any
        # other parent interface.
        log_dir = self.get_log_dir() or self.find_in_context("log_dir")

        # constructing the optimizer.
        optimizer = self.get_optimizer()(
            model.parameters(),
            *self.get_optimizer_args(),
            **self.get_optimizer_kwargs()
        )

        # constructing the scheduler.
        scheduler = (
            self.get_scheduler()
            if self.get_scheduler() is None
            else self.get_scheduler()(
                optimizer, *self.get_scheduler_args(), **self.get_scheduler_kwargs()
            )
        )

        # constructing the classifier.
        return ClassificationSolver(
            model=model.to(self.get_device()),
            log_dir=log_dir,
            epochs=self.get_epochs(),
            train_loader=self.get_train_loader(),
            test_loader=self.get_test_loader(),
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=self.get_loss_fn(),
            device=self.get_device(),
            use_tqdm=self.get_use_tqdm(),
            max_to_keep=self.get_max_checkpoints_to_keep() or "all",
        )
