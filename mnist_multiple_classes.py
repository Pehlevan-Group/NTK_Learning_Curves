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
import scipy.optimize
import numpy as npo
from mnist import MNIST
import pandas as pd
import learning_curves
from jax.config import config
import tensorflow as tf
import tensorflow.keras
import sys
from mpl_toolkits.mplot3d import Axes3D



# set precision
config.update("jax_enable_x64", True)

(images,labels), (x_test,y_test) = tf.keras.datasets.mnist.load_data()

print(images.shape)
images = npo.reshape(images, (images.shape[0], images.shape[1]**2))
x_test = npo.reshape(x_test, (x_test.shape[0], x_test.shape[1]**2))

print(images.shape)
print(x_test.shape)
print(labels.shape)
print(y_test.shape)



# set as numpy array
images = npo.array(images)
labels = npo.array(labels)
x_test = npo.array(x_test)
y_test = npo.array(y_test)


d = 784
num_classes = 10



y_test_mat = npo.zeros((len(y_test), num_classes))
y_train_mat = npo.zeros( (len(labels), num_classes) )

for i in range(len(labels)):
    y_train_mat[i,labels[i]] = 1
for i in range(len(y_test)):
    y_test_mat[i,y_test[i]] = 1


ptot = len(labels)

labels = y_train_mat
y_test = y_test_mat



d = 784
M = 800

# create 3 layer NN and NTK model
init_fn, apply_fn, kernel_fn = stax.serial(
          stax.Dense(M), stax.Relu(),
          stax.Dense(M), stax.Relu(),
          stax.Dense(M), stax.Relu(),
          stax.Dense(num_classes)
      )



key = random.PRNGKey(10)

_, params = init_fn(key, (-1,d))


opt_init, opt_update, get_params = optimizers.adam(2e-3)
nn_loss = jit(lambda params, X, y: np.mean( (apply_fn(params, X) -y)**2 ) )
grad_loss = jit(lambda state, x, y: grad(nn_loss)(get_params(state), x, y))




def train_network(train_set, opt_state, num_iter):

    train_loss = []
    for i in range(num_iter):
        opt_state = opt_update(i, grad_loss(opt_state, *train_set), opt_state )
        loss_i = nn_loss(get_params(opt_state), *train_set)
        train_loss += [loss_i]
        sys.stdout.write('\r loss: %.7f' % loss_i)
        if loss_i < 1e-4:
            break
    sys.stdout.flush()

    return opt_state, train_loss

def neural_net_gen_expt(X, y, nvals):

    num_repeats = 10
    key = random.PRNGKey(10)
    all_keys = random.split(key, num_repeats)
    all_errs = npo.zeros(len(nvals))
    std = npo.zeros(len(nvals))
    #opt_init, opt_update, get_params = optimizers.adam(2.5e-2)
    for i, n in enumerate(nvals):
        print("n = %d" % n)
        errors = npo.zeros(num_repeats)
        for j in range(num_repeats):
            _, params = init_fn(all_keys[j,:], (-1,784))
            opt_state = opt_init(params)
            inds = npo.random.choice(range(len(y)), size = n, replace = False)
            X_j = X[inds,:]
            y_j = y[inds]

            train_set = (X_j,y_j)
            opt_state, train_loss = train_network(train_set, opt_state, 200)
            yhat = apply_fn(get_params(opt_state), X)
            errors[j] = 1/len(y) * npo.linalg.norm(yhat-y)**2
        all_errs[i] = npo.mean(errors)
        std[i] = npo.std(errors)
        sys.stdout.write(' test loss: %.7f | std: %.7f' % (all_errs[i], std[i]) )
        print(" ")
    return all_errs, std



def kernel_gen_expt():

    n = 100
    errors = []
    test_errors = []
    nvals = [5,10,20,50, 100,200,500,1000]
    #nvals = [200]
    num_repeats = 5
    #num_repeats = 1
    all_test_predictions = npo.zeros(len(y_test))
    for n in nvals:
        error = 0
        error_test = 0
        for i in range(num_repeats):
            inds = npo.random.choice(range(len(labels)), size = n, replace = False)
            yhat = nt.predict.gp_inference(kernel_fn, images[inds,:], labels[inds,:], images, get='ntk', diag_reg = 1e-10, compute_cov=False)
            yhat_test = nt.predict.gp_inference(kernel_fn, images[inds,:], labels[inds,:], x_test, get='ntk', diag_reg = 1e-10, compute_cov=False)
            error += 0.5 * 1/len(labels) * npo.linalg.norm(yhat-labels)**2 / num_repeats
            error_test += 0.5 * 1/len(y_test) * npo.linalg.norm( yhat_test[:,0] - y_test )**2 / num_repeats
            all_test_predictions = yhat_test
            print("largest prediction")
            print(np.amax(np.abs(yhat_test)))
        errors.append(error)
        test_errors.append(error_test)
        print(errors)
        print(test_errors)

    return errors, test_errors

# v is eigenvectors
def kernel_gen_expt2(X, y, nvals, v):

    num_repeats = 20
    all_errs = npo.zeros( (len(nvals), v.shape[1]) )
    std = npo.zeros( (len(nvals), v.shape[1]) )
    mode_agg = npo.zeros((len(nvals), 5))
    mode_agg_std = npo.zeros((len(nvals), 5))

    for i, n in enumerate(nvals):
        print("n = %d" % n)
        errors = npo.zeros( (num_repeats, v.shape[1]) )
        for j in range(num_repeats):
            inds = npo.random.choice(range(len(y)), size = n, replace = False)
            yhat = nt.predict.gp_inference(kernel_fn, X[inds,:], y[inds,:], X, get='ntk', diag_reg = 1e-11, compute_cov=False)
            proj_residual = v.T @ (y-yhat)
            errors[j,:] = 1/len(y) * npo.sum( proj_residual**2, axis = 1)
            total = 1/len(y) * npo.linalg.norm(yhat - y)**2
            diff = npo.sum(errors[j,:]) - total
            print("diff: %.8f" % diff)
            #errors[j] = 1/len(y) * npo.linalg.norm(yhat-y)**2

        all_errs[i,:] = npo.mean(errors, axis = 0)
        std[i,:] = npo.std(errors, axis = 0)
        mode_agg[i,0] = np.mean( np.sum(errors[:,0:100], axis = 1) , axis=0)
        mode_agg[i,1] = np.mean( np.sum(errors[:,100:500], axis = 1), axis=0)
        mode_agg[i,2] = np.mean( np.sum(errors[:,500:1000], axis = 1), axis=0)
        mode_agg[i,3] = np.mean( np.sum(errors[:,1000:5000], axis = 1), axis=0)
        mode_agg[i,4] = np.mean( np.sum(errors[:,1000:5000], axis = 1), axis=0)


        mode_agg_std[i,0] = np.std( np.sum(errors[:,0:100], axis = 1) , axis=0)
        mode_agg_std[i,1] = np.std( np.sum(errors[:,100:500], axis = 1), axis=0)
        mode_agg_std[i,2] = np.std( np.sum(errors[:,500:1000], axis = 1), axis=0)
        mode_agg_std[i,3] = np.std( np.sum(errors[:,1000:5000], axis = 1), axis=0)
        mode_agg_std[i,4] = np.std( np.sum(errors[:,1000:5000], axis = 1), axis=0)

        #mode_agg_std[i,0] = np.std( np.sum(errors[:,0:200], axis = 1) , axis=0)
        #mode_agg_std[i,1] = np.std( np.sum(errors[:,200:1000], axis = 1), axis=0)
        #mode_agg_std[i,2] = np.std( np.sum(errors[:,1000:5000], axis = 1), axis=0)
    return all_errs, std, mode_agg, mode_agg_std

# solve lambda = 0 for convenience
def solve_implicit_negative_moment(pvals, moments, spectrum):
    m1 = npo.sum(spectrum)
    roots = npo.zeros(len(pvals))
    for i in range(len(pvals)):
        p = pvals[i]
        args = (p, moments)
        # find polynomial coefficients!!!!!!
        npo.roots()
        sol = sp.optimize.root_scalar(implicit_equation, fprime = f_prime_imp, x0 = m1, method = 'newton', args = (p,npo.array(moments)))
        roots[i] = sol.root
        print(sol.root)
    return roots

def implicit_equation(t, *args):
    p, moments = args
    z = (-1)*t/p
    z_powers = npo.array( [z**(i) for i in range(len(moments))] )
    return 1 - 1/p * npo.dot(moments, z_powers)

def f_prime_imp(t, *args):
    p, moments = args
    z  = (-1)*t/p
    z_powers = npo.array( [ (i+1) * z**(i+1)/t for i in range(len(moments)-1)] )
    return - 1/p * npo.dot(moments[1:len(moments)], z_powers)


def implicit_fn_true(z,*args):
    (p, lamb, spectrum) = args
    return z - lamb - z * npo.dot(spectrum, (p*spectrum + z*npo.ones(len(spectrum)) )**(-1))

def f_prime_true(z,*args):
    (p, lamb, spectrum) = args
    return 1 - npo.dot(spectrum, (p*spectrum + z*np.ones(len(spectrum)) )**(-1)) + z* npo.dot(spectrum, (p*spectrum + z*npo.ones(len(spectrum)) )**(-2))

def solve_implicit_z(spectrum, pvals, lamb):
    sols = npo.zeros(len(pvals))
    for i, p in enumerate(pvals):
        args = (p, lamb, spectrum)
        sols[i] = sp.optimize.root_scalar(implicit_fn_true, x0= p * npo.amax(spectrum), args = args, fprime = f_prime_true, method = 'newton').root
    return sols

def gamma(spectrum, p,lamb, z):
    return z**2 * npo.dot(spectrum**2, (z*np.ones(len(spectrum)) + spectrum*p)**(-2))

def theory_learning_curves(spectrum, eig_vecs, pvals, lamb, y):
    coeffs = eig_vecs.T @ y
    w = np.diag(spectrum**(-0.5)) @ coeffs
    z_vals = solve_implicit_z(spectrum, pvals, lamb)
    gamma_vals = npo.array( [gamma(spectrum, pvals[i],lamb, z_vals[i]) for i in range(len(pvals))] )
    mode_errs = npo.zeros( (len(pvals),len(spectrum)) )
    for i,lambda_rho in enumerate(spectrum):
        mode_errs[:,i] = np.sum(w[i,:]**2)/lambda_rho * z_vals**2 /(z_vals**2 - gamma_vals*pvals) * lambda_rho**2 * z_vals**2 / (lambda_rho*pvals + z_vals)**2
    return mode_errs


# kernel generalization experiment
errors, test_errors = kernel_gen_expt()
df = pd.DataFrame(errors)
df.to_csv('mnist_train_errs_expt.csv', header = None)

nvals = [5,10,20,50, 100,200,500,1000]
plt.loglog(nvals,errors, label = 'training data')
plt.loglog(nvals, test_errors, label = 'test set')
#plt.loglog(nvals, test_errors, label = 'test set')
plt.legend()
plt.xlabel(r'$p$')
plt.ylabel(r'$E_g$')
plt.tight_layout()
plt.savefig('expt_kernel_regression_mnist_01.pdf')
plt.show()


npo.random.seed(100)

plt.rcParams.update({'font.size': 12})


# kernel PCA
num_pca = 8000
inds = npo.random.choice(images.shape[0], size = num_pca, replace = False)
X = images[inds,:]
y = labels[inds,:]
K = kernel_fn(X,X, get='ntk')
print("getting eigenspectrum")
spectrum, vecs = npo.linalg.eigh(1/num_pca * K)
sort_inds = npo.argsort(spectrum)[::-1]
#nvals = npo.logspace(1, np.log10(0.5*num_pca), 7).astype('int')
nvals = npo.logspace(1, np.log10(0.3*num_pca), 6).astype('int')

# sort the spectrum and vectors
spectrum = spectrum[sort_inds]
vecs = vecs[:,sort_inds]


plt.loglog(spectrum)
plt.xlabel(r'$k$')
plt.ylabel(r'$\lambda_k$')
plt.savefig('NTK_MNIST_spectrum.pdf')
plt.show()


df_spec = pd.DataFrame(spectrum)
df_spec.to_csv('MNIST_spectrum_depth3.csv')

eig_vecs = vecs[:,sort_inds]
coeffs = eig_vecs.T @ y
w_teach = coeffs
df_teach = pd.DataFrame(w_teach**2)
df_teach.to_csv('MNIST_teacher_spectrum_depth3.csv')



lamb = 1e-10
print("len of spectrum")
print(len(spectrum))
pvals = npo.logspace(np.log10(10), np.log10(num_pca-1), 500)
sols = solve_implicit_z(npo.array(spectrum), pvals, lamb)

mode_errs = theory_learning_curves(spectrum, vecs, pvals, lamb, y)
sort_inds = np.argsort(spectrum)[::-1]


theory0 = npo.sum(mode_errs, axis = 1)
theory_adj = npo.sum(mode_errs, axis = 1) * num_pca / (num_pca - pvals + 1e-3)
plt.loglog(pvals, theory0, label = 'original theory')
plt.loglog(pvals, theory_adj, label = 'rescaled theory')
plt.legend()
plt.ylim([np.amin(theory_adj), np.amax(theory_adj)])
plt.xlabel(r'$p$', fontsize = 20)
plt.ylabel(r'$E_g$', fontsize=20)
plt.tight_layout()
plt.savefig('rescale_risk.pdf')
plt.show()

inds = [10, 100, 1000]


for i, j in enumerate(sort_inds[inds]):
    if inds[i]==0:
        plt.loglog(pvals, mode_errs[:,j] / mode_errs[0,j], label = r'$k=0$')
    else:
        plt.loglog(pvals, mode_errs[:,j] / mode_errs[0,j], label = r'$k = 10^{%d}$' % int(np.log10(inds[i])+0.01) )

plt.legend()
plt.xlabel(r'$p$', fontsize= 20)
plt.ylabel(r'$E_{k}(p) / E_{k}(0)$', fontsize =20)
plt.tight_layout()
plt.savefig('theory_mode_errs_mnist_3layer.pdf')
plt.show()



# train NN on least-squares objective
nn_errors, nn_std = neural_net_gen_expt(X,y,nvals)
print("finished NN expt")

# get mode errors from kernel expt
expt_mode_errors, expt_mode_std, mode_agg, mode_agg_std = kernel_gen_expt2(X,y,nvals, vecs)



plt.plot( np.log10(pvals) , np.log10( 1/num_pca * npo.sum(mode_errs[:,0:100], axis=1) ), color='C0', label = 'k=1-100')
plt.plot( np.log10(pvals),  np.log10(1/num_pca *npo.sum(mode_errs[:,100:500], axis=1) ), color = 'C1', label = 'k=101-500')
plt.plot( np.log10(pvals), np.log10( 1/num_pca *npo.sum(mode_errs[:,500:1000], axis=1) ) , color = 'C2', label = 'k=501-1000')
plt.plot( np.log10(pvals), np.log10( 1/num_pca *npo.sum(mode_errs[:,1000:5000], axis=1) ) , color = 'C3', label = 'k=1001-5000')
#plt.plot( np.log10(pvals), np.log10( 1/num_pca *npo.sum(mode_errs[:,5000:5000], axis=1) ) , color = 'C4', label = 'k=1001-5000')

plt.errorbar( np.log10(nvals), np.log10(mode_agg[:,0]), mode_agg_std[:,0] / mode_agg[:,0], fmt='^', color='C0')
plt.errorbar( np.log10(nvals), np.log10(mode_agg[:,1]), mode_agg_std[:,1] / mode_agg[:,1], fmt='^', color='C1')
plt.errorbar( np.log10(nvals), np.log10(mode_agg[:,2]), mode_agg_std[:,2] / mode_agg[:,2], fmt='^', color = 'C2')
plt.errorbar( np.log10(nvals), np.log10(mode_agg[:,3]), mode_agg_std[:,3] / mode_agg[:,3], fmt='^', color = 'C3')
#plt.errorbar( np.log10(nvals), np.log10(mode_agg[:,4]), mode_agg_std[:,4] / mode_agg[:,4], fmt='^', color = 'C4')

plt.xticks([1,2,3], [r'$10^{1}$', r'$10^{2}$', r'$10^{3}$'], fontsize=16)
plt.yticks([0,-1,-2,-3], [r'$10^{0}$', r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$'], fontsize=16)
plt.xlabel(r'$p$', fontsize=20)
plt.ylabel(r'$E_g$', fontsize=20)
plt.legend()
plt.tight_layout()
plt.savefig('mode_errs_log_scale.pdf')
plt.show()


theory_agg = npo.zeros((len(pvals), 3))
theory_agg[:,0]= np.sum(mode_errs[:,0:10])
theory_agg[:,1]= np.sum(mode_errs[:,10:100])
theory_agg[:,2]= np.sum(mode_errs[:,100:1000])

for i in range(3):
    plt.loglog(nvals, mode_agg[:,i])
    plt.loglog(pvals, theory_agg[:,i])
plt.show()



pd.DataFrame(expt_mode_errors).to_csv('mnist_mode.csv')
pd.DataFrame(expt_mode_std).to_csv('mnist_std.csv')


inds = [1,10,100, 1000]

rescale = mode_errs[0,sort_inds[1]] / expt_mode_errors[0, sort_inds[1]]
expt_mode_errors = rescale* expt_mode_errors
for i, j in enumerate(sort_inds[inds]):
    plt.plot( np.log10(pvals), np.log10(mode_errs[:,j]), label = r'$k = 10^{%d}$' % int(np.log10(inds[i])+0.01) , color = 'C%d' % i)
    plt.errorbar( np.log10(nvals), np.log10(expt_mode_errors[:,j]), expt_mode_std[:,j] / expt_mode_errors[:,j] * rescale ,fmt ='o', color= 'C%d' % i)
    #plt.plot(pvals, mode_errs[:,j] / mode_errs[0,j])
    #plt.plot(nvals, expt_mode_errors[:,j] / expt_mode_errors[0,j], 'o')
plt.legend()
plt.yticks([3,2,1,0,-1,-2], [r'$10^{0}$', r'$10^{-1}$', r'$10^{-2}$', r'$10^{-3}$', r'$10^{-4}$', r'$10^{-5}$'], fontsize=16)
plt.xticks([1,2,3], [r'$10^{1}$', r'$10^{2}$', r'$10^{3}$'], fontsize=16)
plt.xlabel(r'$p$', fontsize= 20)
plt.ylabel(r'$E_{k}(p)$', fontsize =20)
plt.tight_layout()
plt.savefig('theory_expt_mode_errs_mnist_3layer.pdf')
plt.show()

print("finished kernel expt")
plt.loglog(nvals, errors)
plt.loglog(nvals, nn_errors)
plt.show()

#nn_errors = nn_errors/ nn_errors[0] * errors[0]
#nn_std = nn_std / nn_errors[0] * errors[0]

all_expt_data = npo.zeros((5,len(nvals)))
all_expt_data[0,:] = nvals
all_expt_data[1,:] = errors
all_expt_data[2,:] = std
all_expt_data[3,:] = nn_errors / num_pca
all_expt_data[4,:] = nn_std / num_pca


df = pd.DataFrame(all_expt_data)
df.to_csv('mnist_expt_data_M%d.csv' % M)




#spectrum = spectrum[sort_inds]
#spectrum = spectrum[::-1]
#print(spectrum)
#pve = npo.zeros(len(spectrum)-1)
#for i in range(len(spectrum)-1):
#    pve[i] = npo.sum(spectrum[0:i+1]) / npo.sum(spectrum)

#plt.loglog(spectrum)
#plt.savefig('spectrum_random_subsample.pdf')
#plt.show()




theory = np.sum(mode_errs, axis = 1) / num_pca

theory_vals = npo.zeros((2,len(theory)))
theory_vals[0,:] = pvals
theory_vals[1,:] = theory
df_theory = pd.DataFrame(theory_vals)
df_theory.to_csv('mnist_theory_M%d.csv' % M)

plt.plot( npo.log10(pvals), npo.log10(theory), label = 'MNIST Theory', color = 'C0')
plt.errorbar( npo.log10(nvals), npo.log10(errors), std/errors, fmt = '^', label = 'kernel', color = 'C0')
plt.errorbar( npo.log10(nvals), npo.log10(nn_errors), nn_std/nn_errors, fmt='o', label = 'NN', color = 'C0')
#plt.errorbar( npo.log10(nvals), npo.log10(errors), std/errors, fmt = 'o', label = 'MNIST Expt', color = 'C0')
plt.yticks([-2,-1,0,1], [r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$'], fontsize=16)
plt.xticks([1,2,3], [r'$10^{1}$', r'$10^{2}$', r'$10^{3}$'], fontsize=16)
plt.xlabel(r'$p$', fontsize=20)
plt.ylabel(r'$E_g$', fontsize=20)
plt.legend()
plt.tight_layout()
plt.savefig('MNIST_expt_vs_theory_3_layer_nn.pdf')
plt.show()
