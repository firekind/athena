# Project Athena

Project Athena is a simple wrapper aroung pytorch lightning that helps in quickly defining experiments around a deep learning model.

## Installation

Make sure pytorch version 1.6.0 or any other compatible version is installed, along with torchvision version 0.7.0. Then run,

```
$ pip install git+https://github.com/firekind/athena
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

from athena import datasets, Experiment, ClassificationSolver
from athena.models import MnistNet

# defining batch size and device
batch_size = 128 if torch.cuda.is_available() else 64

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
```