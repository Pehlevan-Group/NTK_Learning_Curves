import numpy as np
from jax import random, jit
import jax.numpy as jnp
import scipy as sp
import scipy.special
import scipy.stats
from ntk_generalization.utils import gegenbauer


# draws random points on unit sphere
def sample_random_points(num_pts, d, key):
    R = random.normal(key, (num_pts, d))
    R = R / jnp.sqrt( jnp.sum(R**2, axis = 1) )[:,jnp.newaxis]
    return R

geg_jit = jit(gegenbauer.get_gegenbauer_lax)

# Kernel is K = \sum_k lambda_k * Qk(z)
def compute_kernel(X, Xp, spectrum, d, kvals):
    P = X.shape[0]
    Pp = Xp.shape[0]
    gram = X @ Xp.T
    gram = jnp.reshape(gram, P*Pp)
    Q = geg_jit(gram, kvals, d)
    degens = jnp.array( [get_degeneracy(d,k) for k in range(kmax)] )
    #K = Q.T @ (spectrum * degens)
    K = Q.T @ spectrum
    K = jnp.reshape(K, (P,Pp))
    return K


# Fit polynomials of the form y = Qk(beta * x) for k=k_target
def mode_err_expt(P, k_target, spectrum, kmax, d, num_repeats, lamb = 1e-6):
  beta = 1.0/jnp.sqrt(d) * random.normal(random.PRNGKey(0), (d,))
  beta = beta / jnp.sqrt(jnp.sum(beta**2))
  key = random.PRNGKey(0)
  all_errs = np.zeros(num_repeats)
  kvals = jnp.arange(kmax)
  for i in range(num_repeats):
    key_tr, key = random.split(key)
    X = sample_random_points(P, d, key_tr)
    y = geg_jit( X @ beta , kvals , d)[k_target,:]
    G = X @ X.T
    K = compute_kernel(X,X,spectrum, d , kvals)
    alpha = jnp.linalg.solve(K + lamb*jnp.eye(P), y) 

    num_test = 2500
    key_te, key = random.split(key)
    X_test = sample_random_points(num_test, d, key_te)
    Kte = compute_kernel(X_test,X, spectrum, d, kvals)
    f = Kte @ alpha
    y = geg_jit( X_test @ beta , kvals, d )[k_target,:]
    all_errs[i] = jnp.mean((f-y)**2)
  return all_errs


