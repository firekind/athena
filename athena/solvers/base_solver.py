import abc

import torch.nn as nn

class BaseSolver(abc.ABC):
    def __init__(self, model: nn.Module):
        self.model = model
        
    @abc.abstractmethod
    def train(self, *args):
        pass
