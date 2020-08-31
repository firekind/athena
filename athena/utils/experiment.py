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
        """
        Runs the experiment. More specifically, calls the `BaseSolver.train` method and saves the 
        `History`.
        """
        
        flush: bool = self.train_args.get("use_tqdm", False)
        print("\033[1m\033[92m=> Running experiment: %s\033[0m" % self.name, flush=flush)

        self.history = self.solver.train(**self.train_args)
