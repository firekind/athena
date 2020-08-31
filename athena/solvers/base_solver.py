import abc

import torch.nn as nn

class BaseSolver(abc.ABC):
    def __init__(self, model: nn.Module):
        """
        The base class for a ``Solver``.

        Args:
            model (nn.Module): The model the solver should act on.
        """
        
        self.model = model
        
    @abc.abstractmethod
    def train(self, *args):
        pass
