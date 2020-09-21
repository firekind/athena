# Project Athena

Project Athena is a python package which helps in experimenting with various iterations of a deep learning model. The core of this package is the `Experiment` object, which contains the information about an experiment, and also the information obtained while training.

## Installation

Make sure pytorch version 1.6.0 or any other compatible version is installed, along with torchvision version 0.7.0. Then run,

```
$ pip install git+https://github.com/firekind/athena@v0.0.1
```
to install the package.

## Development

To set up the development environment, clone the repo and run

```
$ make venv
```

to make the virtual environment and install an editable version of athena. Then install pytorch version 1.6.0 and torchvision version 0.7.0 in the virtual environment.

## Usage

```python
# importing
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from athena import datasets, Experiments, ClassificationSolver
from athena.models import MnistNet

# defining batch size and device
batch_size = 128 if torch.cuda.is_available() else 64
device = "cuda" if torch.cuda.is_available() else "cpu"

# creating the datasets 
train_loader = (
    datasets.mnist()
    .batch_size(batch_size)
    .use_default_transforms()
    .build()
)

test_loader = (
    datasets.mnist()
    .test()
    .batch_size(batch_size)
    .use_default_transforms()
    .build()
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
    .build()
)

# running experiment
exp.run()
```

To run multiple experiments one after the other, the `Experiments` class is used.

```python
...
from athena import Experiments
...

exps = (
    Experiments("MNIST experiments)
    .log_directory("./logs") # optional. if not given, tensorboard will not be used.
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
        .build()
    .done()
)

exps.run()
```

You can specify a custom loss function to use as well, for example:

```python
...
def custom_loss_fn(y_pred, y_true):
    y_pred = F.log_softmax(y_pred)
    return F.nll_loss(y_pred, y_true)

exp = (
    Experiment("ResNet with custom loss function")
    .model(ResNet32())
    .solver(ClassificationSolver)
        .optimizer(optim.SGD, lr=0.01, momentum=0.9)
        .scheduler(StepLR, step_size=8, gamma=0.1)
        .epochs(epochs)
        .train_loader(train_loader)
        .test_loader(test_loader)
        .loss_fn(custom_loss_fn) # specifying loss function to use
        .device(device)
        .build()
    .build()
)

exp.run()
```