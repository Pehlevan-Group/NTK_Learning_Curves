
# Learning Curves for NTK and Wide Neural Networks
See our preprint on [Arxiv](https://arxiv.org/abs/2002.02561)

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

`M` is the number of hidden units in each layer. Depth limited to 2,3,4.

### Two-Layer NN with Composite Targets

`python two_layer.py --input_dim [d] --M [M]`

### Plotting

`python plot_two_layer.py --input_dim [d] --M [M]`

`python plot_curve.py`

Change the file name in plot_curve.py to plot the results from the Pure Mode experiments.
Some examples are stored in the results directory.
