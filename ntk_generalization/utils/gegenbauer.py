import numpy as np
import jax.numpy as jnp
from jax import lax, jit
from scipy.special import comb
import scipy.special as spe
import scipy as sp
import scipy.misc
import math
#import numba
#from numba import jit


# ratio of sphere surface area in dim d to d-1
def area_ratio(dim):
    return np.sqrt(np.pi)*np.exp(spe.gammaln((dim-1)/2) - spe.gammaln((dim)/2))


# jax.lax recursion for fast gegenbauer
def get_gegenbauer_lax(x,kvals,d):
    alpha = d/2-1.0
    def recurse_Q(Qk_Qkm, km):
      k = km + 2
      Qk,Qkm = Qk_Qkm
      Qnew = 1/k * ( 2 * x*(k + alpha -1) *Qk - (k + 2*alpha - 2) * Qkm )
      return [Qnew,Qk], Qkm

    Q0 = jnp.ones(x.shape[0])
    Q1 = 2*alpha*x
    _, Q = lax.scan(recurse_Q,[Q1,Q0], kvals)
    return Q

# standard for loop gegenbauer recursion
def gegenbauer_loop(x, kvals, d):
    alpha = d / 2.0 - 1
    kmax = len(kvals)
    Q = np.zeros((kmax, len(x)))
    Q[0, :] = np.ones(len(x))
    Q[1, :] = 2 * alpha * x
    for k_m in range(kmax - 2):
        k = k_m + 2
        Q[k, :] = 1 / k * (2 * x * (k + alpha - 1) * Q[k - 1, :] - (k + 2 * alpha - 2) * Q[k - 2, :])
    return Q

# switching to Mathematica convention
def get_degeneracy(d, k):
    alpha = (d - 2)/2.0
    return (k + alpha) / alpha * comb(k + 2 * alpha - 1, k)


def eigenvalue_normalization(kmax, alpha):
    
    dim = 2.0*alpha + 2.0
    ## Implements omega(D)/omega(D-1) in the paper
    area_ratio = np.sqrt(np.pi)*np.exp(spe.gammaln((dim-1)/2) - spe.gammaln((dim)/2))
    
    norm_factor = np.zeros(kmax)
    for k in range(kmax):
        norm_factor[k] = area_ratio * (alpha / (alpha + k))

    return norm_factor
