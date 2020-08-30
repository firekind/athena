from typing import Dict, Any

import torch.nn as nn

from . import History
from athena.solvers import BaseSolver


class Experiment:
    def __init__(
        self,
        name: str,
        model: nn.Module,
        train_args: Dict[str, Any],
        solver_cls: BaseSolver,
    ):
        self.name = name
        self.model = model
        self.train_args = train_args
        self.history: History = None
        self.solver = solver_cls(model)

    def run(self):
        print("=> Running experiment:", self.name)
        self.history = self.solver.train(**self.train_args)
