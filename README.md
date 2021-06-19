# Learning Curves for NTK and Wide Neural Networks
See our preprint on [Arxiv](https://arxiv.org/abs/2002.02561)

## Setup

Using the [Python dependency manager `poetry`](https://python-poetry.org/) you can easily install all dependencies for this project with:
```
$ poetry install
```

Now start the virtual environment (and run all subsequent commands within it):
```
$ poetry shell
```

## Experiments

### Kernel Regression experiment with NTK
To generate experimental and theoretical learning curves for kernel regression with ReLU NTK run

`python kernel_regression_lc.py --input_dim [d] --lamb [lamb] --NTK_depth [depth]`

The optional arguments are: `input_dim` is the dimension of the data,
`lamb` is the explicit regularizer,
`NTK_depth` is the number of layers for the fully connected ReLU NTK.
These parameters default to the values used in the paper.

### NN with Pure Mode Targets

`python NTK_gd_comparison.py --input_dim [d] --M [M] --depth [depth] --learning_rage [lr]`

`M` is the number of hidden units in each layer.

### Two-Layer NN with Composite Targets

`python two_layer.py --input_dim [d] --M [M]`


### MNIST Experiments with NTK and NN regression

`python mnist_multiple_classes.py`

### Plotting

`python plot_two_layer.py --input_dim [d] --M [M]`

`python plot_curve.py`

Change the file name in plot_curve.py to plot the results from the Pure Mode experiments.
Some examples are stored in the results directory.
