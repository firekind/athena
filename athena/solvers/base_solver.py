import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import Optimizer, _LRScheduler


class BaseSolver(pl.LightningModule):
    def __init__(
        self, model: nn.Module, optimizer: Optimizer, scheduler: _LRScheduler = None
    ):
        """
        The base class for all the solvers. A solver is just a pytorch lightning ``LightningModule``
        with a single constraint: The model, optimizer and scheduler has to be passed as constructor
        arguments (in the same order and the same spelling, if you want to use the ``Experiment`` builder
        API to define experiments)

        This class implements the ``forward`` and ``configure_optimizers`` function of the ``LightningModule``.

        Args:
            model (nn.Module): The model to use.
            optimizer (Optimizer): The optimizer to use.
            scheduler (_LRScheduler, optional): The scheduler to use. Defaults to None.
        """

        super(BaseSolver, self).__init__()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def configure_optimizers(self):
        if self.scheduler is None:
            return self.optimizer

        return [self.optimizer], [self.scheduler]
