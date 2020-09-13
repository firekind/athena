
Welcome to Athena's documentation!
==================================
.. toctree::
   :hidden:
   :glob:
   :maxdepth: 2
   :caption: Contents
   
   apis/athena
   apis/athena.datasets
   apis/athena.layers
   apis/athena.models
   apis/athena.solvers
   apis/athena.utils


Athena is a package which was developed during the Extensive Vision AI Program of the School Of AI. This package is a helper package which
is used to rapidly experiment during the assignments of the program. Although this package was made during the program, it can be used in other situations as well.


Usage
-----

The core of this package is the :class:`athena.utils.experiment.Experiment` class, which is used to quickly set up an experiment with 
various parameters that are to be used for the experiment.

.. code-block:: python

   # importing
   import torch.optim as optim
   from torch.optim.lr_scheduler import StepLR

   from athena import datasets, Experiments, ClassificationSolver
   from athena.models import MnistNet

   # defining batch size and device
   batch_size = 128 if torch.cuda.is_available() else 64
   device = "cuda" if torch.cuda.is_available() else "cpu"

   # creating the datasets 
   train_loader = datasets.mnist(
      batch_size=batch_size,
      use_default_transforms=True,
   )

   test_loader = datasets.mnist(
      train=False,
      batch_size=batch_size,
      use_default_transforms=True,
   )

   # creating the experiment
   exp = (
      Experiment("Ghost batch norm with 2 splits")
      .model(MnistNet(use_ghost_batch_norm=True))
      .solver(ClassificationSolver)
      .optimizer(optim.SGD, lr=0.01, momentum=0.9)
      .scheduler(StepLR, step_size=8, gamma=0.1)
      .epochs(epochs)
      .train_loader(train_loader)
      .test_loader(test_loader)
      .device(device)
      .build()
   )

   # running experiment
   exp.run()

To run mulitple experiments one after the other, the :class:`athena.utils.experiment.Experiments` class is used.

.. code-block:: python

   ...
   from athena import Experiments
   ...

   exps = (
      Experiments()
      .add("Ghost batch norm with 2 splits")
         .model(MnistNet(use_ghost_batch_norm=True))
         .solver(ClassificationSolver)
         .optimizer(optim.SGD, lr=0.01, momentum=0.9)
         .scheduler(StepLR, step_size=8, gamma=0.1)
         .epochs(epochs)
         .train_loader(train_loader)
         .test_loader(test_loader)
         .device(device)
         .build()

      .add("Ghost batch norm with 4 splits")
         .model(MnistNet(use_ghost_batch_norm=True))
         .solver(ClassificationSolver)
         .optimizer(optim.SGD, lr=0.01, momentum=0.9)
         .scheduler(StepLR, step_size=8, gamma=0.1)
         .epochs(epochs)
         .train_loader(train_loader)
         .test_loader(test_loader)
         .device(device)
         .build()
      .done()
   )

   exps.run()

Index
-----
* :ref:`genindex`
