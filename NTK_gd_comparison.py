import jax
import jax.numpy as np
from jax import random
from jax.experimental import optimizers
from jax.api import jit, grad, vmap
import functools
import neural_tangents as nt
from neural_tangents import stax
import matplotlib.pyplot as plt
import scipy as sp
import scipy.special
import gegenbauer
import pandas as pd
import numpy as npo
import argparse

def target_fn(x,target_pts,alpha, k, d):
    p1 = x.shape[0]
    p2 = x.shape[0]
    gram = np.matmul(x,target_pts.T)
    Q = gegenbauer.get_gegenbauer_fast2( gram , k+2, d)[k,:]
    y = npo.matmul(Q, alpha)
    return y

def sample_random_points(num_pts, d):
    R = npo.random.multivariate_normal(np.zeros(d), np.eye(d), num_pts)
    R = R* npo.outer( npo.linalg.norm(R, axis=1)**(-1), np.ones(d) )
    return np.array(R)

def get_datasets(ensemble_size, train_points, test_points, num_targets, d, k):

    train_x = npo.zeros((ensemble_size, train_points, d))
    train_y = npo.zeros((ensemble_size, train_points, 1))
    test_x = npo.zeros((ensemble_size, test_points, d))
    test_y = npo.zeros((ensemble_size, test_points, 1))

    for i in range(ensemble_size):
        target_pts = sample_random_points(num_targets, d)
        alpha = random.normal(target_pt_key, (num_targets, 1))
        train_xs = sample_random_points(train_points, d)
        train_ys = target_fn(train_xs, target_pts, alpha, k, d)

        test_xs = sample_random_points(test_points,d)
        test_ys = target_fn(test_xs, target_pts, alpha, k, d)

        train_x[i,:,:] = train_xs
        train_y[i,:,:] = train_ys
        test_x[i,:,:] = test_xs
        test_y[i,:,:] = test_ys

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    return train_x, train_y, test_x, test_y


parser = argparse.ArgumentParser()
parser.add_argument('--input_dim', type=int, default= 30,
  help='data input dimension')
parser.add_argument('--M', type=int,
  help='number of hidden units', default = 500)
parser.add_argument('--depth', type=int, default= 2,
  help='network depth; integer')
parser.add_argument('--num_targets', type=int, default= 10000,
  help='number of points in ')
parser.add_argument('--test_points', type=int, default= 1000,
  help='number of test points to estimate generalization error')
parser.add_argument('--learning_rate', type=float, default= 0.25,
  help='learning rate')
parser.add_argument('--training_steps', type=int, default= 10000,
  help='number of test points to estimate generalization error')

args = parser.parse_args()
d = args.input_dim
M = args.M
depth = args.depth
num_targets = args.num_targets
test_points = args.test_points
learning_rate = args.learning_rate
training_steps = args.training_steps

def create_network(depth, width):
    layers = []
    for l in range(depth):
        layers += [stax.Dense(M, W_std=1.5, b_std=0.0, parameterization ='ntk'), stax.Relu()]
    layers += [stax.Dense(1,W_std=1.5, b_std = 0, parameterization = 'ntk')]
    return stax.serial(*layers)

init_fn, apply_fn, kernel_fn = stax.serial(depth, M)

apply_fn = jit(apply_fn)
kernel_fn = jit(kernel_fn, static_argnums=(2,))

key = random.PRNGKey(10)

key, net_key = random.split(key)
_, params = init_fn(net_key, (-1,d))



key, x_key, y_key, test_key, target_pt_key = random.split(key, 5)

opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
opt_update = jit(opt_update)

loss = jit(lambda params, x, y: 0.5 * np.mean((apply_fn(params, x) - y) ** 2))
grad_loss = jit(lambda state, x, y: grad(loss)(get_params(state), x, y))
loss_kernel = jit(lambda yhat, y: 0.5  * np.mean( (yhat - y) ** 2))

def train_network(key, trainx, trainy, testx, testy):

  train = (trainx, trainy)
  test = (testx, testy)

  train_losses = []
  test_losses = []

  _, params = init_fn(key, (-1, d))
  opt_state = opt_init(params)
  batchsize = min(50, trainy.shape[0])
  for i in range(training_steps):
    randinds = npo.random.randint(0,trainy.shape[0],batchsize)
    data_i = (trainx[randinds,:],trainy[randinds])
    train_i = np.reshape(loss(get_params(opt_state), *data_i), (1,))
    train_losses += [train_i]
    opt_state = opt_update(i, grad_loss(opt_state, *data_i), opt_state)

  train_losses = np.concatenate(train_losses)
  test_loss = np.reshape(loss(get_params(opt_state), *test), (1,))
  return get_params(opt_state), train_losses, test_loss


def NTK_test_err(key, trainx, trainy, testx, testy):
    y_hat = nt.predict.gp_inference(kernel_fn, trainx, trainy, testx, diag_reg=1e-10, get='ntk', compute_cov = False)
    return loss_kernel(y_hat, testy)





ensemble_size = 30

kvals = [1,2,4]
pvals = [5,10,20,50,100, 250, 500,1000]

all_mean_test_losses = []
all_std_test_losses = []
all_kernel_test = []
train_k = []
all_std_kernel_test = []
for k in kvals:
    mean_test_p = []
    std_test_p = []
    kernel_test_p = []
    std_kernel_p = []
    for i in range(len(pvals)):
        train_points = pvals[i]

        ensemble_key = random.split(key, ensemble_size)
        kvals = k*np.ones(ensemble_size).astype('int')
        trainx, trainy, testx, testy = get_datasets(ensemble_size, train_points, test_points, num_targets, d,k)
        params, train_loss, test_loss = vmap(train_network, (0,0,0,0,0), 0)(ensemble_key, trainx, trainy, testx, testy)
        kernel_test = vmap(NTK_test_err, (0,0,0,0,0), 0)(ensemble_key, trainx, trainy, testx, testy)
        mean_test_p.append( npo.nanmean(test_loss) )
        std_test_p.append( npo.nanstd(test_loss) )
        kernel_test_p.append( npo.nanmean(kernel_test) )
        std_kernel_p.append( npo.nanstd(kernel_test) )


    all_mean_test_losses.append(mean_test_p)
    all_std_test_losses.append(std_test_p)
    all_kernel_test.append(kernel_test_p)
    all_std_kernel_test.append( std_kernel_p )
    train_k.append( np.mean(train_loss, axis=0) )
all_mean_test_losses = npo.array(all_mean_test_losses)
all_std_test_losses = npo.array(all_std_test_losses)
all_kernel_test = npo.array(all_kernel_test)
all_std_kernel_test = npo.array(all_std_kernel_test)
train_errs = npo.array(train_k)

meandf = pd.DataFrame(all_mean_test_losses)
stddf = pd.DataFrame(all_std_test_losses)
kerneldf = pd.DataFrame(all_kernel_test)
kernel_std_df = pd.DataFrame(all_std_kernel_test)
traindf = pd.DataFrame(train_errs)

meandf.to_csv('results/all_test_losses_rough_4_layer_d%d_M%d_target%d_numiter%d_lr%d.csv' % (d,M, num_targets, training_steps,learning_rate))
stddf.to_csv('results/std_test_losses_rough_4_layer_d%d_M%d_target%d_numiter%d_lr%d.csv' % (d,M, num_targets, training_steps,learning_rate))
kerneldf.to_csv('results/all_kernel_test_rough_4_layer_d%d_M%d_target%d_numiter%d_lr%d.csv' % (d,M, num_targets, training_steps, learning_rate))
traindf.to_csv('results/train_loss_4_layer_%d_M%d_target%d_numiter%d_lr%d.csv' %(d,M, num_targets, training_steps,learning_rate))
kernel_std_df.to_csv('results/all_kernel_std_test_rough_4_layer_d%d_M%d_target%d_numiter%d_lr%d.csv' % (d,M, num_targets, training_steps, learning_rate))
