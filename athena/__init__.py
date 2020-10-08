import logging
from .experiment import Experiment, Experiments
from .solvers import ClassificationSolver

logging.getLogger("lightning").setLevel(logging.ERROR)