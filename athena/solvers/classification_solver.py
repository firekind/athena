from typing import Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import Optimizer, _LRScheduler
import torch.nn.functional as F

from .base_solver import BaseSolver


class ClassificationSolver(BaseSolver):
    def __init__(self, model: nn.Module, optimizer: Optimizer, scheduler: _LRScheduler=None, loss_fn:Callable=None):
        """
        Solver for classification problems.

        Args:
            model (nn.Module): The model to use.
            optimizer (Optimizer): The optimizer to use.
            scheduler (_LRScheduler, optional): The scheduler to use. Defaults to ``None``.
            loss_fn (Callable, optional): The loss function to use. Defaults to ``None``. \
                if ``None``, cross entropy loss will be used.

        """
        super(ClassificationSolver, self).__init__(model, optimizer, scheduler)

        self.loss_fn = F.cross_entropy if loss_fn is None else loss_fn
        self.running_train_acc = 0

    def on_train_epoch_start(self):
        self.running_train_acc = 0

    def training_step(self, batch, batch_idx):
        data, target = batch
        y_pred = self(data)

        loss = self.loss_fn(y_pred, target)
        acc = self.acc_fn(y_pred, target)
        self.running_train_acc += acc

        avg_running_acc = self.running_train_acc / (batch_idx + 1)

        return {
            "loss": loss,
            "acc": avg_running_acc,
            "progress_bar": {"train accuracy": avg_running_acc},
        }

    def training_epoch_end(self, outputs):
        loss = torch.mean(torch.tensor([o["loss"] for o in outputs]))
        acc = outputs[-1]["acc"]

        return {
            "log": {
                "training loss": loss,
                "training accuracy": acc,
                "step": self.current_epoch + 1,
            }
        }

    def validation_step(self, batch, batch_idx):
        data, target = batch
        y_pred = self(data)

        loss = self.loss_fn(y_pred, target)
        acc = self.acc_fn(y_pred, target)

        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        acc = torch.tensor([o["val_acc"] for o in outputs]).mean()
        loss = torch.tensor([o["val_loss"] for o in outputs]).mean()

        results = pl.EvalResult(checkpoint_on=loss)
        results.log("validation loss", loss, prog_bar=True)
        results.log("validation accuracy", acc, prog_bar=True)
        results.log("step", self.current_epoch + 1)

        return results

    def acc_fn(self, outputs, targets):
        pred_classes = outputs.detach().argmax(dim=1, keepdim=True)
        correct = pred_classes.eq(targets.view_as(pred_classes)).float()

        return correct.mean()
