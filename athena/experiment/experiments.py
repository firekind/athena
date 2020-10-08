import os
from typing import Dict, Optional

from .experiment import Experiment, ExperimentBuilder


class Experiments:
    def __init__(self, name: str, experiments: Dict, log_dir: str = None):
        """
        Defines experiments that has to be run one after the other.

        Args:
            name (str): The name of the list of experiments.
            experiments (Dict): The name-experiment key value pairs.
            log_dir (str, optional): The parent of the directory where the logs should be stored. \
                the logs will be stored in the directory ``os.path.join(log_dir, name)``.
        """

        self.name = name
        self.experiments = experiments
        self.log_dir = log_dir

    def run(self):
        """
        Runs all the experiments.
        """

        for _, exp in self.experiments.items():
            exp.run()

    def __getitem__(self, name):
        return self.experiments[name]

    def __iter__(self):
        for name, obj in self.experiments.items():
            yield obj

    @staticmethod
    def builder() -> "ExperimentsBuilder":
        """
        Returns an object of the builder interface. Needs to be called if one wants
        to use the builder pattern to define the ``Experiments``.

        Returns:
            ExperimentsBuilder: The builder interface for ``Experiments``.
        """
        return ExperimentsBuilder()


class ExperimentsBuilder:
    def __init__(self):
        """
        A Builder interface for the :class:`Experiments`.
        """

        self.experiments = {}
        self.props = {}

    def add(self, name) -> ExperimentBuilder:
        """
        Adds an experiment to the list.

        Args:
            name (str): The name of the experiment.

        Returns:
            ExperimentBuilder: The builder interface for :class:`Experiment`
        """

        assert (
            self.props.get("name") is not None
        ), "Set the name of this group of experiments first."
        assert (
            self.props.get("log_dir") is not None
        ), "Set the log directory for these experiments first."

        return ExperimentBuilder(name, self)

    def build(self) -> Experiments:
        """
        Creates and returns an object of :class:`Experiments`.

        Returns:
            Experiments
        """

        # assertion checks
        assert (
            self.props.get("name") is not None
        ), "Set the name of this group of experiments."
        assert self.props.get("log_dir") is not None, "Set the log directory."

        # creating `Experiments` object.
        return Experiments(self.props["name"], self.experiments, self.props["log_dir"])

    def handle(self, obj: object):
        """
        Handles objects built by the ``build`` method of :class:`ExperimentBuilder`.
        This function should be explicitly called in the ``build`` method of the :class:`ExperimentBuilder`.

        Args:
            obj (object): The object built by the ``build`` method of :class:`ExperimentBuilder`.
        """
        if isinstance(obj, Experiment):
            self.experiments[obj.name] = obj

    def name(self, name: str):
        """
        Sets the name of the group of experiments.

        Args:
            name (str): The name of the group.

        Returns:
            ExperimentsBuilder: The object of this class.
        """

        self.props["name"] = name
        return self

    def log_directory(self, path) -> "ExperimentsBuilder":
        """
        Sets the log directory of the experiment.

        Args:
            path (str): The path to the directory.

        Returns:
            ExperimentsBuilder: Object of this class.
        """
        assert (
            self.props.get("name") is not None
        ), "Set the name of this group of experiments before setting log directory."

        self.props["log_dir"] = os.path.join(path, self.props["name"])

        return self

    def get_log_dir(self) -> Optional[str]:
        """
        Getter for the log directory.

        Returns:
            Optional[str]: The path to the log directory as a string, or \
                ``None`` if no log directory is set.
        """
        return self.props.get("log_dir")
