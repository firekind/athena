athena.experiment package
=========================

.. toctree::
   :hidden:

Core classes behind the athena package.

The :class:`athena.experiment.Experiment` class wraps around a :class:`athena.solvers.base_solver.BaseSolver`, and provides methods to run, visualize and tune an experiment.

The :class:`athena.experiment.ExperimentBuilder` provides a builder style interface to create experiment(s). There are three sections to be called while using the class:

* ``props()``: This section defines the properties of the experiment, such as name, log directory, trainer arguments and so on.
* ``data()``: This section defines the dataloaders that will be used for the experiment.
* ``solver(cls)``: This section defines the solver and the properties of the solver.

After defining all the section, the experiment is compiled using the ``build()`` function.

.. code-block:: python

   # creating the experiment
   exp = (
      Experiment.builder()
      .props() # the properties section
         .name("MNIST with ghost batch norm with 2 splits")
         .log_directory("./logs")
      .data() # the data section
         .train_loader(train_loader)
         .val_loader(test_loader)
      .solver(ClassificationSolver) # the solver section
         .epochs(10)
         .model(MnistNet(use_ghost_batch_norm=True))
         .optimizer(optim.SGD, lr=0.01, momentum=0.9)
         .scheduler(StepLR, step_size=8, gamma=0.1)
      .build() # compiling the experiment
   )

The :class:`athena.experiment.ExperimentBuilder` makes it easy to use custom created solvers in the builder api. under the ``solver(cls)`` section, any arguments that are defined in the constructor of the solver can be accessed as functions in the ``solver(cls)`` section.

.. code-block:: python

   # creating a custom solver
   class MyCustomSolver(BaseSolver):
      def __init__(model, optimizer, scheduler, custom_value_1, custom_value_2):
         super(MyCustomSolver, self).__init__(model, optimizer, scheduler)
         
         # do stuff with custom_value_1 and custom_value_2

      # defining remaining methods (except forward and configure_optimizers) just like in pytorch lightning

   .
   .
   .

   exp = (
      Experiment.builder()
      .props()
         .name("Using MyCustomSolver")
         .log_directory("./logs")
      .data()
         # add data loaders
      .solver(MyCustomSolver)
         .epochs(10)
         .model(...)
         .optimizer(...)
         .scheduler(...)
         .custom_value_1(...) # custom_value_1 can be given a value like this
         .custom_value_2(...) # custom_value_2 can be given a value like this
      .build()
   )
   
   .
   .
   .

.. automodule:: athena.experiment
   :members:
   :undoc-members:
   :show-inheritance:
