import os
import json
from pathlib import Path
from datetime import datetime
from typing import Union, List, Dict
from abc import ABC, abstractmethod

import torch


class Checkpoint:
    def __init__(self, path: str, track: List, max_to_keep: Union[int, str] = "all"):
        """
        Creates, restores and manages checkpoint files.

        Args:
            path (str): The path to the checkpoint directory.
            track (List): a list of objects that either extends :class:`Checkpointable` or a \
                valid pytorch object that has ``state_dict`` and ``load_state_dict`` attributes.
            max_to_keep (Union[int, str], optional): The max number of checkpoint files to keep. \
                Defaults to "all".
        """

        self.path = Path(path)
        self.objects: List = track

        self.index = _Index(str(self.path), max_to_keep)

    def save(self):
        """
        Creates a checkpoint file.
        """

        # create dict to save
        data: Dict = {}
        for i, obj in enumerate(self.objects):
            data[i] = obj.state_dict()

        # save dict
        torch.save(data, self.current_checkpoint_path)

        # updating state
        to_remove = self.index.add_checkpoint(self.current_checkpoint_path, datetime.now())

        # delete old
        self._delete_checkpoint(to_remove)

    def restore(self, path: str = None):
        """
        Restores data from a checkpoint.

        Args:
            path (str, optional): The path to the checkpoint file. If None, restores from the \
                latest checkpoint. Defaults to None.
        """
        # getting the latest checkpoint
        checkpoint_path = path or self.index.get_latest_checkpoint()

        # if there is no latest checkpoint, returning
        if checkpoint_path is None:
            return

        # else loading it and restoring the objects
        for idx, value in torch.load(checkpoint_path).items():
            self.objects[idx].load_state_dict(value)

        print(f"Restored checkpoint from {checkpoint_path}")

    def _delete_checkpoint(self, path: str):
        """
        Deletes a checkpoint file.

        Args:
            path (str): Path to the checkpoint file.
        """

        if path is None:
            return

        os.remove(path)

    @property
    def current_checkpoint_path(self):
        """
        Path to the current checkpoint.
        """
        return os.path.join(self.path, f"checkpoint_{self.index.get_file_count() + 1}")


class _Index:
    def __init__(self, path: str, max_to_keep: Union[str, int]):
        """
        Represents all the required information about the checkpoints.

        Args:
            path (str): The path to the directory that will contain the checkpoints.
            max_to_keep (Union[str, int]): The max number of checkpoints to keep.
        """

        self.parent_dir = Path(path)
        self.path = Path(os.path.join(path, ".index"))
        self.max_to_keep = max_to_keep

        self.created_on = None
        self.timestamp_format = "%d-%m-%Y %H:%M:%S.%f"
        self.file_count = 0
        self.checkpoints = []

        self.load()

    def load(self):
        """
        Creates or loads the info from the index file.
        """

        try:
            # creating the directory
            self.parent_dir.mkdir(parents=True)

            # creating the index file
            with open(self.path, "w") as f:
                json.dump(
                    {
                        "max_to_keep": self.max_to_keep,
                        "created_on": datetime.now().strftime(self.timestamp_format),
                        "file_count": self.file_count,
                        "checkpoints": self.checkpoints,
                    },
                    f,
                )

        except FileExistsError:
            # loading the data in the index file
            with open(self.path, "r") as f:
                data = json.load(f)

            # updating instance variables from file
            self.max_to_keep = data["max_to_keep"]
            self.created_on = datetime.strptime(
                data["created_on"], self.timestamp_format
            )
            self.file_count = data["file_count"]
            self.checkpoints = data["checkpoints"]

    def add_checkpoint(self, checkpoint_file_path: str, timestamp: datetime) -> str:
        """
        Updates the index file.

        Args:
            checkpoint_file_path (str): The path to the checkpoint file that has to be added \
                to the index file
            timestamp (datetime): The timestamp of when the checkpoint file was created.

        Returns:
            str: The old checkpoint file to remove.
        """

        # incrementing file count
        self.increment_file_count()

        # getting the index file contents
        with open(self.path, "r") as f:
            data = json.load(f)

        # updating the file count
        data["file_count"] = self.file_count

        # getting checkpoint files present
        files = data["checkpoints"]

        # removing old files
        if self.max_to_keep != "all" and len(files) >= self.max_to_keep:
            to_remove = files.pop(0)
        else:
            to_remove = None

        # adding the current checkpoint file
        files.append(self._create_checkpoint_entry(checkpoint_file_path, timestamp))
        data["checkpoints"] = files

        # writing contents to checkpoint file
        with open(self.path, "w") as f:
            json.dump(data, f)

        # returning path of the file to remove or None if there
        # is nothing to remove
        if to_remove is None:
            return None
        return to_remove["path"]

    def get_latest_checkpoint(self) -> str:
        """
        Gets the path to the latest checkpoint.

        Returns:
            str:
        """
        # loading index file
        with open(self.path, "r") as f:
            data = json.load(f)

        try:
            # returning the path of the last checkpoint entry
            return data["checkpoints"][-1]["path"]
        except IndexError:
            # at this stage there are no checkpoint entries, so returning None
            return None

    def increment_file_count(self):
        """
        Increments the file count by 1.
        """
        self.file_count += 1

    def _create_checkpoint_entry(self, path: str, timestamp: datetime):
        """
        Creates a checkpoint entry for the index file.

        Args:
            path (str): The path to the checkpoint file.
            timestamp (datetime): The timestamp.

        Returns:
            Dict: The entry.
        """
        return {
            "path": path,
            "timestamp": datetime.strftime(timestamp, self.timestamp_format),
        }

    def get_file_count(self) -> int:
        """
        Getter for the file count.

        Returns:
            int
        """
        return self.file_count


class Checkpointable(ABC):
    @abstractmethod
    def load_state_dict(self, data: Dict):
        """
        Loads the state of the object from the given data. This data is the same \
            data that is returned from :func:`state_dict`.

        Args:
            data (Dict): The data from the checkpoint file.
        """

    @abstractmethod
    def state_dict(self) -> Dict:
        """
        Returns the data in the instance that has to be checkpointed.

        Returns:
            Dict: A dictionary containing the data to be checkpointed.
        """