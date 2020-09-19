from typing import Callable, List, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR, Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from athena.utils import History

from .base_solver import BaseSolver, BatchResult, StepResult


class ClassificationSolver(BaseSolver):
    def __init__(
        self,
        model: nn.Module,
        epochs: int = None,
        train_loader: DataLoader = None,
        test_loader: DataLoader = None,
        optimizer: Optimizer = None,
        scheduler: LRScheduler = None,
        loss_fn: Callable = None,
        device: str = None,
        use_tqdm: bool = False,
        log_dir: str = None,
    ):
        """
        A solver for classification problems.

        Args:
            model (nn.Module): The model to act on.
            epochs (int, optional): The number of epochs to train for. Defaults to None.
            train_loader (DataLoader, optional): The ``DataLoader`` for the training data. Defaults to None.
            test_loader (DataLoader, optional): The ``DataLoader`` for the test data. Defaults to None.
            optimizer (Optimizer, optional): The optimizer to use. Defaults to None.
            scheduler (LRScheduler, optional): The ``LRScheduler`` to use. Defaults to None.
            loss_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional): The loss function to use. If not given, model \
                will be trained using negative log likelihood loss with reduction as ``mean``. Defaults to None.
            device (str, optional): A valid pytorch device string. Defaults to None.
            use_tqdm (bool, optional): If True, uses tqdm instead of a keras style progress bar (``pkbar``). Defaults to False.
            log_dir (str, optional): The directory to store the logs. Defaults to None.
        """

        super(ClassificationSolver, self).__init__(
            model=model,
            epochs=epochs,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            use_tqdm=use_tqdm,
            log_dir=log_dir,
        )

    def train(self):
        """
        Trains the model.

        Args:

        """
        # defining variables
        self.history = History()
        if self.get_loss_fn() is None:
            print(
                "\033[1m\033[93mWarning:\033[0m Loss function not specified. Using nll loss.",
                flush=self.should_use_tqdm(),
            )
            self.set_loss_fn(F.nll_loss)

        # adding model to graph
        images, labels = next(iter(self.get_train_loader()))
        self.writer_add_model(self.get_model(), images.to(self.get_device()))

        # training
        for self.current_epoch in range(self.get_epochs()):
            print(
                "Epoch: %d / %d" % (self.current_epoch + 1, self.get_epochs()),
                flush=self.should_use_tqdm(),
            )

            # performing train step
            self.train_step()

            # stepping scheduler
            if self.get_scheduler() is not None and not isinstance(
                self.get_scheduler(), OneCycleLR
            ):
                self.get_scheduler().step()

            # performing test step
            if self.get_test_loader() is not None:
                self.test_step()

        self.writer_close()

    @BaseSolver.log_results
    @BaseSolver.prog_bar()
    def train_step(self) -> List[Tuple[str, float]]:
        """
        Performs a single train step.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple consisting of the average train loss and average
            train accuracy.
        """

        # setting model in train mode
        self.get_model().train()

        # defining variables
        correct = 0
        processed = 0
        train_loss = 0
        for batch_idx, (data, target) in enumerate(self.get_train_loader()):
            res = self.train_on_batch(
                batch=data.to(self.get_device()),
                target=target.to(self.get_device()),
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
                ("loss", float(train_loss / len(self.get_train_loader()))),
                (
                    "accuracy",
                    float(100 * correct / len(self.get_train_loader().dataset)),
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
        self.get_optimizer().zero_grad()

        # forward prop
        y_pred = self.get_model()(batch)

        # calculating loss
        loss = self.get_loss_fn()(y_pred, target)
        running_train_loss += loss.detach()

        # backpropagation
        loss.backward()
        self.get_optimizer().step()

        # calculating accuracy
        pred = y_pred.argmax(dim=1, keepdim=True)
        running_correct += pred.eq(target.view_as(pred)).sum().item()
        running_processed += len(batch)
        acc = 100 * running_correct / running_processed

        if isinstance(self.get_scheduler(), OneCycleLR):
            self.get_scheduler().step()

        return BatchResult(
            batch_idx=batch_idx,
            data=[("loss", loss.item()), ("accuracy", acc)],
            running_correct=running_correct,
            running_processed=running_processed,
            running_train_loss=running_train_loss,
        )

    @BaseSolver.log_results
    def test_step(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a single test step.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple consisting of the average test loss and average test accuracy.
        """

        # setting model to evaluation mode
        self.get_model().eval()

        # defining variables
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.get_test_loader():
                # casting data to device
                data, target = data.to(self.get_device()), target.to(self.get_device())

                # forward prop
                output = self.get_model()(data)

                # calculating loss
                test_loss += self.get_loss_fn()(output, target)

                # calculating number of correctly predicted classes
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        # averaging loss
        test_loss /= len(self.get_test_loader())
        test_acc = 100.0 * correct / len(self.get_test_loader().dataset)

        # printing result
        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss,
                correct,
                len(self.get_test_loader().dataset),
                test_acc,
            ),
            flush=self.should_use_tqdm(),
        )

        return StepResult(data=[("test loss", test_loss), ("test accuracy", test_acc)])

    def get_misclassified(
        self, data_loader: DataLoader, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gets the information of the misclassified data items in the given dataset.

        Args:
            test_loader (DataLoader): The ``DataLoader`` to use.
            device (str): A valid pytorch device string.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple consisting of the image information, predicted, and actual class of the misclassified images.
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
                output = self.get_model()(data)

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
