from typing import Tuple
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from .progbar import ProgbarCallback
from .transforms import ToNumpy, ToTensor, UnNormalize


def get_misclassified(
    module: pl.LightningModule, data_loader: DataLoader, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
        Gets the information of the misclassified data items in the given dataset.

        Args:
            module (pl.LightningModule): The module to use.
            test_loader (DataLoader): The ``DataLoader`` to use.
            device (str): A valid pytorch device string.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple consisting of the \
                image information, predicted, and actual class of the misclassified images.
        """

    # defining variables
    misclassified = []
    misclassified_pred = []
    misclassified_target = []

    module.to(device)

    # put the model to evaluation mode
    module.eval()

    with torch.no_grad():
        for data, target in data_loader:
            # casting data to device
            data, target = data.to(device), target.to(device)

            # forward prop
            output = module(data)

            # get the predicted class
            pred = output.argmax(dim=1, keepdim=True)

            # get the current misclassified in this batch
            list_misclassified = pred.eq(target.view_as(pred)) == False
            batch_misclassified = data[list_misclassified.squeeze()]
            batch_mis_pred = pred[list_misclassified]
            batch_mis_target = target.view_as(pred)[list_misclassified]

            # add data to function variables
            misclassified.append(batch_misclassified)
            misclassified_pred.append(batch_mis_pred)
            misclassified_target.append(batch_mis_target)

    # group all the batched together
    misclassified = torch.cat(misclassified)
    misclassified_pred = torch.cat(misclassified_pred)
    misclassified_target = torch.cat(misclassified_target)

    return misclassified, misclassified_pred, misclassified_target
