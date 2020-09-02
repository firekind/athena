from typing import List, Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
import numpy as np
from tqdm import tqdm
from pkbar import Kbar

from athena.utils import History
from .base_solver import BaseSolver


class RegressionSolver(BaseSolver):
    def __init__(self, model: nn.Module):
        """
        A solver for regression problems. This solver supports models that have mulitple inputs, and thus, 
        multiple losses

        Args:
            model (nn.Module): The model to act on.
        """

        super(RegressionSolver, self).__init__(model)

    def train(
        self,
        epochs: int,
        train_loader: DataLoader,
        optimizer: Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], List[Tuple[str, torch.Tensor]]],
        acc_fn: Callable[
            [List[Tuple[str, torch.Tensor]], torch.Tensor],
            List[Tuple[str, torch.Tensor]],
        ],
        scheduler: LRScheduler = None,
        test_loader: DataLoader = None,
        device: str = "cpu",
        use_tqdm: bool = False,
    ) -> History:
        """
        Trains the model.

            **Note**: look at the notes in :meth:`athena.solvers.regression_solver.RegressionSolver.train_step` and in \
                :meth:`athena.solvers.regression_solver.RegressionSolver.test_step`.

        Args:
            epochs (int): The number of epochs to train for.
            train_loader (DataLoader): The ``DataLoader`` for the training data.
            optimizer (Optimizer): The optimizer to use.
            loss_fn (Callable[[torch.Tensor, torch.Tensor], List[Tuple[str, torch.Tensor]]]): The loss function to use. \
                the loss function should take in the predicted output of the model and target output from the dataset as \
                the arguments and return a list of tuples, in which the first element of each tuple is a label for the \
                loss and the second element is the loss value.
            acc_fn (Callable[[List[Tuple[str, torch.Tensor]], torch.Tensor], List[Tuple[str, torch.Tensor]]]): The accuracy \
                function to use. The function should take in two arguments, first, a list of tuples, where the first element \ 
                of each tuple is the label for the loss and the second element is the loss value, and the second argument \
                the target output from the dataset. The function should return a list of tuples, first element of the tuple \
                should be the label of the accuracy and the second element should be the accuracy value.
            scheduler (LRScheduler, optional): The ``LRScheduler`` to use. Defaults to None.
            test_loader (DataLoader, optional): The ``DataLoader`` for the test data. Defaults to None.
            device (str, optional): A valid pytorch device string. Defaults to ``cpu``.
            use_tqdm (bool, optional): If True, uses tqdm instead of a keras style progress bar (``pkbar``). Defaults to False.

        Returns:
            History: An History object containing training information.
        """
        history = History()

        for epoch in range(epochs):
            print("Epoch: %d / %d" % (epoch + 1, epochs), flush=use_tqdm)

            # performing train step
            train_data = self.train_step(
                train_loader, optimizer, scheduler, device, loss_fn, use_tqdm
            )

            # adding metrics to history
            for label, data in train_data:
                history.add_metric(label, data)

            # stepping scheduler
            if scheduler is not None and not isinstance(scheduler, OneCycleLR):
                scheduler.step()

            # performing test step
            if test_loader is not None:
                test_data = self.test_step(
                    test_loader, device, loss_fn, flush_print=use_tqdm
                )

                # adding metrics to history
                for label, data in test_data:
                    history.add_metric(label, data)

        return history

    def train_step(
        self,
        train_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        device: str,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], List[Tuple[str, torch.Tensor]]],
        acc_fn: Callable[
            [List[Tuple[str, torch.Tensor]], torch.Tensor],
            List[Tuple[str, torch.Tensor]],
        ],
        use_tqdm: bool,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Performs a single train step.
            
            **Note**: the losses and accuracies returned by the ``loss_fn`` and ``acc_fn`` are divided by the \
                number of batches in the dataset while recording them for an epoch (averaging). So make \
                sure any reduction in your functions are 'mean'.
            
        Args:
            train_loader (DataLoader): The ``DataLoader`` for the training data.
            optimizer (Optimizer): The optimizer to use.
            scheduler (LRScheduler): The LR scheduler to use.
            device (str): A valid pytorch device string.
            loss_fn (Callable[[torch.Tensor, torch.Tensor], List[Tuple[str, torch.Tensor]]]): The loss function to use. \
                the loss function should take in the predicted output of the model and target output from the dataset as \
                the arguments and return a list of tuples, in which the first element of each tuple is a label for the \
                loss and the second element is the loss value.
            acc_fn (Callable[[List[Tuple[str, torch.Tensor]], torch.Tensor], List[Tuple[str, torch.Tensor]]]): The accuracy \
                function to use. The function should take in two arguments, first, a list of tuples, where the first element \ 
                of each tuple is the label for the loss and the second element is the loss value, and the second argument \
                the target output from the dataset. The function should return a list of tuples, first element of the tuple \
                should be the label of the accuracy and the second element should be the accuracy value.
            use_tqdm (bool): If True, uses tqdm instead of a keras style progress bar (``pkbar``).

        Returns:
            List[Tuple[str, torch.Tensor]]: A list containing tuples in which the first element of the tuple is the label \
                describing the value and the second element is the value itself.
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
        train_losses: np.ndarray = None
        train_accs: np.ndarray = None
        for batch_idx, (data, target) in enumerate(iterator):
            # casting to device
            data, target = data.to(device), target.to(device)

            # zeroing out accumulated gradients
            optimizer.zero_grad()

            # forward prop
            y_pred = self.model(data)

            # calculating loss (look at function documentation for details on what is returned by
            # the loss_fn)
            losses_data: List[Tuple[str, torch.Tensor]] = loss_fn(y_pred, target)
            if train_losses is None:
                train_losses = np.fromiter(
                    [x[-1] for x in losses_data], dtype=np.float32
                )
            else:
                train_losses = train_losses + np.fromiter(
                    [x[-1] for x in losses_data], dtype=np.float32
                )

            # backpropagation
            for _, loss in losses_data:
                loss.backward()
            optimizer.step()

            # calculating the accuracies (look at function documentation for details on what is returned by
            # the acc_fn)
            acc_data: List[Tuple[str, torch.Tensor]] = acc_fn(losses_data, target)
            if train_accs is None:
                train_accs = np.fromiter([x[-1] for x in acc_data], dtype=np.float32)
            else:
                train_accs = train_accs + np.fromiter(
                    [x[-1] for x in acc_data], dtype=np.float32
                )

            # updating progress bar with instantaneous losses and accuracies
            if use_tqdm:
                losses_desc = " - ".join(
                    [f"{name}: {value:0.4f}" for name, value in losses_data]
                )
                accs_desc = " - ".join(
                    [f"{name}: {value:0.4f}" for name, value in acc_data]
                )
                pbar.set_description(
                    desc=f"Batch_id: {batch_idx + 1} - {losses_desc} - {accs_desc}"
                )
            else:
                pbar.update(batch_idx, values=[*losses_data, *acc_data])

            if isinstance(scheduler, OneCycleLR):
                scheduler.step()

        if not use_tqdm:
            # required for pkbar
            pbar.add(1, values=[*losses_data, *acc_data])

        return [
            *list(
                zip(
                    # getting the labels of each loss value
                    [x[0] for x in losses_data],
                    # dividing the value of each of the losses by the number of batches in the dataset
                    [loss / len(train_loader) for loss in train_losses],
                )
            ),
            *list(
                zip(
                    # getting the labels of each accuracy value
                    [x[0] for x in acc_data],
                    # dividing the value of each accuracy by the number of batches in the dataset
                    [acc / len(train_loader) for acc in train_accs],
                )
            ),
        ]

    def test_step(
        self,
        test_loader: DataLoader,
        device: str,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], List[Tuple[str, torch.Tensor]]],
        acc_fn: Callable[
            [List[Tuple[str, torch.Tensor]], torch.Tensor],
            List[Tuple[str, torch.Tensor]],
        ],
        flush_print: bool = False,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Performs a single test step.

            **Note**: the losses and accuracies returned by the ``loss_fn`` and ``acc_fn`` are divided by the \
            number of batches in the dataset (while calculating the average loss and average accuracy during the \
            train step) and displayed as the result. So make sure any reduction in your functions are 'mean'.

        Args:
            test_loader (DataLoader): The ``DataLoader`` for the test data.
            device (str): A valid pytorch device string.
            loss_fn (Callable[[torch.Tensor, torch.Tensor], List[Tuple[str, torch.Tensor]]]): The loss function to use. \
                the loss function should take in the predicted output of the model and target output from the dataset as \
                the arguments and return a list of tuples, in which the first element of each tuple is a label for the \
                loss and the second element is the loss value.
            acc_fn (Callable[[List[Tuple[str, torch.Tensor]], torch.Tensor], List[Tuple[str, torch.Tensor]]]): The accuracy \
                function to use. The function should take in two arguments, first, a list of tuples, where the first element \ 
                of each tuple is the label for the loss and the second element is the loss value, and the second argument \
                the target output from the dataset. The function should return a list of tuples, first element of the tuple \
                should be the label of the accuracy and the second element should be the accuracy value.
            flush_print (bool, optional): Whether to flush the print statement or not. Needed when tqdm is used in a notebook. \ 
                Defaults to False.

        Returns:
            List[Tuple[str, torch.Tensor]]: A list containing tuples in which the first element of the tuple is the label \
                describing the value and the second element is the value itself.
        """

        # setting model to evaluation mode
        self.model.eval()

        # defining variables
        test_losses: np.ndarray = None
        test_accs: np.ndarray = None
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                # casting data to device
                data, target = data.to(device), target.to(device)

                # forward prop
                output = self.model(data)

                # calculating loss
                losses_data: List[Tuple[str, torch.Tensor]] = loss_fn(output, target)
                if test_losses is None:
                    test_losses = np.fromiter(
                        [x[-1] for x in losses_data], dtype=np.float32
                    )
                else:
                    test_losses = test_losses + np.fromiter(
                        [x[-1] for x in losses_data], dtype=np.float32
                    )

                # calculating accuracy
                accs_data: List[Tuple[str, torch.Tensor]] = acc_fn(losses_data, target)
                if test_accs is None:
                    test_accs = np.fromiter(
                        [x[-1] for x in accs_data], dtype=np.float32
                    )
                else:
                    test_accs = test_accs + np.fromiter(
                        [x[-1] for x in accs_data], dtype=np.float32
                    )

        # averaging loss
        test_losses /= len(test_loader)
        test_accs /= len(test_loader)

        # constructing average loss data and description to be displayed
        avg_loss_data = []
        for data, loss in zip(losses_data, test_losses):
            avg_loss_data.append((data[0], loss))
        loss_desc = ",".join([f"{name}: {value}" for name, value in avg_loss_data])

        # constructing accuracy data and description to be displayed
        avg_acc_data = []
        for data, acc in zip(accs_data, test_accs):
            avg_acc_data.append((data[0], acc))
        acc_desc = ",".join([f"{name}: {value}" for name, value in avg_acc_data])

        # printing result
        print(f"Test set: {loss_desc}, {acc_desc}", flush=flush_print)

        return [*avg_loss_data, *avg_acc_data]
