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
```

For more details, have a look at the [documentation](firekind.github.io/athena)