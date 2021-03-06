
Welcome to Athena's documentation!
==================================
.. toctree::
   :hidden:
   :glob:
   :maxdepth: 2
   :caption: Contents
   
   apis/athena
   apis/athena.datasets
   apis/athena.experiment
   apis/athena.layers
   apis/athena.models
   apis/athena.solvers
   apis/athena.tuning
   apis/athena.utils
   apis/athena.visualizations

Athena is a package which was developed during the Extensive Vision AI Program of the School Of AI. This package is a helper package which
is used to rapidly experiment during the assignments of the program. Although this package was made during the program, it can be used in other situations as well.


Usage
-----

The core of this package is the :class:`athena.experiment.Experiment` class, which is used to quickly set up an experiment with 
various parameters that are to be used for the experiment.

.. code-block:: python

   # importing
   import torch.optim as optim
   from torch.optim.lr_scheduler import StepLR

   from athena import datasets, Experiment, ClassificationSolver
   from athena.models import MnistNet

   # defining batch size and device
   batch_size = 128 if torch.cuda.is_available() else 64

   # creating the datasets 
   train_loader = (
      datasets.mnist.builder()
      .batch_size(batch_size)
      .use_default_transforms()
      .build()
   )

   test_loader = (
      datasets.mnist.builder()
      .test()
      .batch_size(batch_size)
      .use_default_transforms()
      .build()
   )

   # creating the experiment
   exp = (
      Experiment.builder()
      .props()
         .name("MNIST with ghost batch norm with 2 splits")
         .log_directory("./logs")
      .data()
         .train_loader(train_loader)
         .val_loader(test_loader)
      .solver(ClassificationSolver)
         .epochs(10)
         .model(MnistNet(use_ghost_batch_norm=True))
         .optimizer(optim.SGD, lr=0.01, momentum=0.9)
         .scheduler(StepLR, step_size=8, gamma=0.1)
      .build()
   )

   # running experiment
   exp.run()

To run multiple experiments one after the other, the :class:`athena.utils.experiment.Experiments` class is used.

.. code-block:: python

   ...
   from athena import Experiments
   ...

   exps = (
      Experiments.builder()
      .name("MNIST experiments)
      .log_directory("./logs")
      .add("Ghost batch norm with 2 splits")
         .data()
            .train_loader(train_loader)
            .val_loader(test_loader)
         .solver(ClassificationSolver)
            .epochs(epochs)
            .model(MnistNet(use_ghost_batch_norm=True))
            .optimizer(optim.SGD, lr=0.01, momentum=0.9)
            .scheduler(StepLR, step_size=8, gamma=0.1)
         .build()
      .add("Ghost batch norm with 4 splits")
         .data()
            .train_loader(train_loader)
            .val_loader(test_loader)
         .solver(ClassificationSolver)
            .epochs(epochs)
            .model(MnistNet(use_ghost_batch_norm=True, num_splits=4))
            .optimizer(optim.SGD, lr=0.01, momentum=0.9)
            .scheduler(StepLR, step_size=8, gamma=0.1)
         .build()
      .build()
   )

   exps.run()

You can specify a custom loss function to use as well, for example:

.. code-block:: python

   ...
   def custom_loss_fn(y_pred, y_true):
      y_pred = F.log_softmax(y_pred)
      return F.nll_loss(y_pred, y_true)

   exp = (
      Experiment.builder()
      .props()
         .name("ResNet with custom loss function")
         .log_directory("./logs")
      .data()
         .train_loader(train_loader)
         .val_loader(test_loader)
      .solver(ClassificationSolver)
         .epochs(epochs)
         .model(MnistNet(use_ghost_batch_norm=True))
         .optimizer(optim.SGD, lr=0.01, momentum=0.9)
         .scheduler(StepLR, step_size=8, gamma=0.1)
         .loss_fn(custom_loss_fn)
      .build()
   )

   exp.run()

Index
-----
* :ref:`genindex`
