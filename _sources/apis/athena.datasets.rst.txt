athena.datasets package
=======================

.. toctree::
   :hidden:

Some commonly used datasets.

``athena.datasets.mnist`` module
--------------------------------

The good old MNIST dataset. This module contains a ``builder()`` function to build a ``DataLoader`` of the MNIST dataset, as well as some default train and test transforms that are commonly applied.

.. automodule:: athena.datasets.mnist
   :members:
   :undoc-members:
   :show-inheritance:


``athena.datasets.cifar10`` module
----------------------------------

The CIFAR10 dataset. This module contains a a ``builder()`` function to build a ``DataLoader`` of the CIFAR10 dataset, as well as some default train and test transforms that are commonly applied.

.. automodule:: athena.datasets.cifar10
   :members:
   :undoc-members:
   :show-inheritance:

``athena.datasets.base`` module
---------------------------------------

The base class which all datasets that want to use the builder API must implement.

.. automodule:: athena.datasets.base
   :members:
   :undoc-members:
   :show-inheritance:
