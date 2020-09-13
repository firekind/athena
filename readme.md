# Project Athena

Project Athena is a python package which helps in experimenting with various iterations of a deep learning model. The core of this package is the `Experiment` object, which contains the information about an experiment, and also the information obtained while training.

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

from athena import datasets, Experiments, ClassificationSolver
from athena.models import MnistNet

# defining batch size and device
batch_size = 128 if torch.cuda.is_available() else 64
device = "cuda" if torch.cuda.is_available() else "cpu"

# creating the datasets 
train_loader = datasets.mnist(
    download=True,
    batch_size=batch_size,
    use_default_transforms=True,
)

test_loader = datasets.mnist(
    download=True,
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
```

For more details, have a look at the [documentation](https://firekind.github.io/athena)
