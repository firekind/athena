from typing import List, Dict

from collections import defaultdict
import torch

from athena.utils import Checkpointable

class History(Checkpointable):
    def __init__(self):
        """
        Contains the information of the losses and accuracies that are recorded during training.
        """

        self.data: Dict[str, List[torch.Tensor]] = defaultdict(list)

    def add_metric(self, name: str, value: torch.Tensor):
        """
        Adds an item to the history.

        Args:
            name (str): The name of the data to be added, eg. loss
            value (torch.Tensor): The value to be added.
        """

        self.data[name].append(value)

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
            "data": self.data
        }

    def load_state_dict(self, data: Dict):
        """
        Loads the state from the checkpoint.

        Args:
            data (Dict): The checkpoint data
        """
        self.data = data["data"]