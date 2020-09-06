
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

The core of this package is the :py:class:`athena.utils.experiment.Experiment` class, which is used to quickly set up an experiment with 
various parameters that are to be used for the experiment.

.. code-block:: python

   # importing
   from athena import dataset, Experiment, ClassificationSolver
   from athena.models import MnistNet
   from athena.transforms import mnist_test_transforms, mnist_train_transforms
   from athena.utils.functions import plot_experiments, plot_misclassified

   # defining batch size and device
   batch_size = 128 if torch.cuda.is_available() else 64
   device = "cuda" if torch.cuda.is_available() else "cpu"

   # creating the datasets 
   train_loader = datasets.MNIST(
      root="./data",
      download=True,
      train=True,
      transform=mnist_train_transforms(),
      batch_size=batch_size
   )

   test_loader = datasets.MNIST(
      root="./data",
      download=True,
      train=False,
      transform=mnist_test_transforms(),
      batch_size=batch_size
   )

   # creating experiment
   model = MnistNet(use_ghost_batch_norm=True).to(device)
   optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
   scheduler = StepLR(optimizer, step_size=8, gamma=0.1)
   exp = Experiment(
      name="With Ghost Batch Norm",
      model=model,
      solver_cls=ClassificationSolver,
      train_args=dict(
         epochs=25,
         train_loader=train_loader,
         test_loader=test_loader,
         optimizer=optimizer,
         scheduler=scheduler,
         device=device,
      )
   )

   # running experiment
   exp.run()

Index
-----
* :ref:`genindex`
