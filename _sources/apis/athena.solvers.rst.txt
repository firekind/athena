athena.solvers package
======================

.. toctree::
   :hidden:

A ``Solver`` is a class which is used to train the model. These classes encapsulate the training regime, which is specific to the problem at hand.
Some of these classes also contain helper methods that are specific to the problem, for example, the ``ClassificationSolver`` which is used to train 
classification problems contains a helper method to get ``n`` misclassified images.

``classification_solver`` module
--------------------------------------------

.. automodule:: athena.solvers.classification_solver
   :members:
   :undoc-members:
   :show-inheritance:

``regression_solver`` module
--------------------------------------------

.. automodule:: athena.solvers.regression_solver
   :members:
   :undoc-members:
   :show-inheritance:
