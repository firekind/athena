from typing import List, Dict

import torch
import numpy as np

from athena.utils import Checkpointable

class History(Checkpointable):
    def __init__(self, size: int):
        """
        Contains the information of the losses and accuracies that are recorded during training.

        Args:
            size (int): The max number of losses and accuracies that will be stored.
        """

        self.data: Dict[str, np.ndarray] = {}
        self.size = size

    def add_metric(self, name: str, value: torch.Tensor, step: int):
        """
        Adds an item to the history.

        Args:
            name (str): The name of the data to be added, eg. loss
            value (torch.Tensor): The value to be added.
            step (int): The step (epoch) at which it is being added.

        Raises:
            ValueError: If ``step`` is greater that ``size``.
        """
        if step >= self.size:
            raise ValueError("step ({}) cannot be greater than size ({})".format(step, self.size))
        
        if name not in self.data:
            self.data[name] = np.zeros((self.size,))

        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()

        self.data[name][step] = value

    def get_metric_names(self) -> List[str]:
        """
        Gets a list of names of the metrics being recorded.

        Returns:
            List[str]: The list of metric names.
        """

        return list(self.data.keys())

    def get_metric(self, metric: str) -> List[torch.Tensor]:
        """
        Gets the value of the given metric.

        Args:
            metric (str): The name of the metric.

        Returns:
            List[torch.Tensor]: The value of the metric.
        """

        return self.data[metric]

    def has_metric(self, metric: str) -> bool:
        """
        Checks whether the given metric is being (or has been) recorded or not.

        Args:
            metric (str): The name of the metric

        Returns:
            bool: True, if it is being (or has been) recorded, False otherwise.
        """

        return metric in self.data

    def state_dict(self):
        """
        Returns a state dict to checkpoint.

        Returns:
            Dict
        """

        return {
            "data": self.data,
            "size": self.size
        }

    def load_state_dict(self, data: Dict):
        """
        Loads the state from the checkpoint.

        Args:
            data (Dict): The checkpoint data
        """
        self.data = data["data"]
        self.size = data["size"]