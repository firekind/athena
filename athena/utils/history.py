from typing import List

import torch


class History:
    def __init__(
        self,
        train_losses: List[torch.Tensor],
        train_accs: List[torch.Tensor],
        test_losses: List[torch.Tensor],
        test_accs: List[torch.Tensor],
    ):  
        """
        Contains the information of the losses and accuracies that are recorded during training.

        Args:
            train_losses (List[torch.Tensor]): The training losses.
            train_accs (List[torch.Tensor]): The training accuracies.
            test_losses (List[torch.Tensor]): The test losses.
            test_accs (List[torch.Tensor]): The test accuracies.
        """
        
        self.test_losses = test_losses
        self.test_accs = test_accs
        self.train_losses = train_losses
        self.train_accs = train_accs