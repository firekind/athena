athena.solvers package
======================

.. toctree::
   :hidden:

A ``Solver`` is a class which is used to train the model. It is just a regular pytorch lightning ``LightingModule``, with a single restriction: The model, optimizer and scheduler must be passed in the constructor, and in the same order. Morever, if the solver is to be used using the :class:`athena.experiment.Experiment` builder, Then the name of the parameters must be "model" for the model, "optimizer" for the optimizer, and "scheduler" for the scheduler.

``classification_solver`` module
--------------------------------------------

A simple solver for classification problems.

.. automodule:: athena.solvers.classification_solver
   :members:
   :undoc-members:
   :show-inheritance:

``base_solver`` module
--------------------------------------------

This class implements the `forward` and `configure_optimizers` method so you don't have to.

.. automodule:: athena.solvers.base_solver 
   :members:
   :undoc-members:
   :show-inheritance:
