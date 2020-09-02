from typing import List, Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from tqdm import tqdm
from pkbar import Kbar

from athena.utils import History
from .base_solver import BaseSolver


class ClassificationSolver(BaseSolver):
    def __init__(self, model: nn.Module):
        """
        A solver for classification problems.

        Args:
            model (nn.Module): The model to act on.
        """

        super(ClassificationSolver, self).__init__(model)

    def train(
        self,
        epochs: int,
        train_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler = None,
        test_loader: DataLoader = None,
        device: str = "cpu",
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        use_tqdm: bool = False,
    ) -> History:
        """
        Trains the model.

        Args:
            epochs (int): The number of epochs to train for.
            train_loader (DataLoader): The ``DataLoader`` for the training data.
            optimizer (Optimizer): The optimizer to use.
            scheduler (LRScheduler, optional): The ``LRscheduler`` to use. Defaults to None.
            test_loader (DataLoader, optional): The ``DataLoader`` for the test data. Defaults to None.
            device (str, optional): A valid pytorch device string. Defaults to ``cpu``.
            loss_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional): The loss function to use. If not given, model \
                will be trained using negative log likelihood loss with reduction as 'mean'
            use_tqdm (bool, optional): If True, uses tqdm instead of a keras style progress bar (``pkbar``). Defaults to False.

        Returns:
            History: An History object containing training information.
        """
        history = History()

        if loss_fn is None:
            loss_fn = F.nll_loss

        for epoch in range(epochs):
            print("Epoch: %d / %d" % (epoch + 1, epochs), flush=use_tqdm)
            avg_train_loss, avg_train_acc = self.train_step(
                train_loader, optimizer, scheduler, device, loss_fn, use_tqdm
            )
            history.add_metric("train loss", avg_train_loss)
            history.add_metric("train accuracy", avg_train_acc)

            if scheduler is not None and not isinstance(scheduler, OneCycleLR):
                scheduler.step()

            if test_loader is not None:
                avg_test_loss, avg_test_acc = self.test_step(
                    test_loader, device, loss_fn, flush_print=use_tqdm
                )
                history.add_metric("test loss", avg_test_loss)
                history.add_metric("test accuracy", avg_test_acc)

        return history

    def train_step(
        self,
        train_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        device: str,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        use_tqdm: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a single train step.

        Args:
            train_loader (DataLoader): The ``DataLoader`` for the training data.
            optimizer (Optimizer): The optimizer to use.
            scheduler (LRScheduler): The LR scheduler to use.
            device (str): A valid pytorch device string.
            loss_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function to use.
            use_tqdm (bool): If True, uses tqdm instead of a keras style progress bar (``pkbar``).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple consisting of the average train loss and average
            train accuracy.
        """

        # setting model in train mode
        self.model.train()

        # creating progress bar
        if use_tqdm:
            pbar = tqdm(train_loader)
            iterator = pbar
        else:
            pbar = Kbar(len(train_loader), stateful_metrics=["loss", "accuracy"])
            iterator = train_loader

        # defining variables
        correct = 0
        processed = 0
        train_loss = 0
        for batch_idx, (data, target) in enumerate(iterator):
            # casting to device
            data, target = data.to(device), target.to(device)

            # zeroing out accumulated gradients
            optimizer.zero_grad()

            # forward prop
            y_pred = self.model(data)

            # calculating loss
            loss = loss_fn(y_pred, target)
            train_loss += loss.detach()

            # backpropagation
            loss.backward()
            optimizer.step()

            # calculating accuracy
            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            acc = 100 * correct / processed

            # updating progress bar
            if use_tqdm:
                pbar.set_description(
                    desc=f"Batch_id: {batch_idx + 1} - Loss: {loss.item():0.4f} - Accuracy: {acc:0.2f}%"
                )
            else:
                pbar.update(
                    batch_idx,
                    values=[
                        ("loss", loss.item()),
                        ("accuracy", acc),
                    ],
                )

            if isinstance(scheduler, OneCycleLR):
                scheduler.step()

        if not use_tqdm:
            pbar.add(
                1,
                values=[("loss", loss.item()), ("accuracy", acc)],
            )

        return (
            train_loss / len(train_loader),
            100 * correct / len(train_loader.dataset),
        )

    def test_step(
        self,
        test_loader: DataLoader,
        device: str,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        flush_print: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a single test step.

        Args:
            test_loader (DataLoader): The ``DataLoader`` for the test data.
            device (str): A valid pytorch device string.
            loss_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The loss function to use.
            flush_print (bool, optional): Whether to flush the print statement or not. Needed when tqdm is used in a notebook. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple consisting of the average test loss and average test accuracy.
        """

        # setting model to evaluation mode
        self.model.eval()

        # defining variables
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                # casting data to device
                data, target = data.to(device), target.to(device)

                # forward prop
                output = self.model(data)

                # calculating loss
                test_loss += loss_fn(output, target)

                # calculating number of correctly predicted classes
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        # averaging loss
        test_loss /= len(test_loader)
        test_acc = 100.0 * correct / len(test_loader.dataset)

        # printing result
        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                test_acc,
            ),
            flush=flush_print,
        )

        return (test_loss, test_acc)

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
        self.model.eval()

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
                batch_misclassified = data[list_misclassified]
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