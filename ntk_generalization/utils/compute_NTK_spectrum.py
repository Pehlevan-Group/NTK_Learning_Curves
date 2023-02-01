import numpy as np
import jax.numpy as jnp
import math
import matplotlib.pyplot as plt

from ntk_generalization.utils import gegenbauer
import scipy as sp
import scipy.special as spe
import scipy.optimize




# compute NTK iteratively
def NTK_iter(z, L):
  # z = x * x' = cos(theta)
  h = 1.0*z
  all_Phi = []
  for l in range(L):   
    h = Phi_kernel(h)
    all_Phi += [h]

  K = jnp.zeros(z.shape[0])
  all_G = [Dot_Phi_kernel(all_Phi[-2])] # < dot_phi(h^L) dot_phi(h^L) >_{h^L ~ Phi^{L-1}}
  for l in range(depth-2):
    all_G.insert(0,  all_G[0] * Dot_Phi_kernel(all_Phi[-3-l]) )
  all_G.insert(0, all_G[0] * Dot_Phi_kernel(z))
  K = all_Phi[-1] + all_G[0] * z
  for l in range(L-1):
    K += all_G[l+1] * all_Phi[l]
  return K


def Phi_kernel(z):
  return 0.5 /jnp.pi * ( jnp.sqrt(1-z**2) + (jnp.pi - jnp.arccos(z))*z )

def Dot_Phi_kernel(z):
    return 0.5/jnp.pi *( jnp.pi - jnp.arccos(z))


def get_effective_spectrum(layers, kmax, d, ker = 'NTK'):

    normed_spec = np.zeros((len(layers), kmax))
    num_pts = 2000
    
    # alpha parameter (see Wolfram Alpha)
    alpha = d/2.0 - 1

    # get roots and weights for gegenbauer quadrature
    z, w = spe.roots_gegenbauer(num_pts, alpha)
    Q_sp = jnp.array([spe.gegenbauer(k,alpha)(z) for k in range(kmax)])
    Q_lax = gegenbauer.get_gegenbauer_lax(z, jnp.arange(kmax),d)
    Q_loop = gegenbauer.gegenbauer_loop(z, jnp.arange(kmax),d)

    # deg k polynomial degeneracy 
    degens = np.array( [gegenbauer.get_degeneracy(d,k) for k in range(kmax)] )
    # normalization coefficient for eigenvalues
    norms = gegenbauer.eigenvalue_normalization(kmax, alpha)

    # compute gegenbauer polynomials over quad. points
    #Q = get_gegenbauer_lax(z, jnp.arange(kmax), d)

    for i,depth in enumerate(layers):
      
      # calculate the NTK on quadrature roots
      NTKz = NTK_iter(z , depth)

      scaled_NTK = NTKz * w
      #scaled_Q = jnp.einsum('km,k->km', Q, degens)
      #scaled_Q =  Q * np.outer(degens, np.ones(num_pts))

      spectrum_scaled = Q_lax @ scaled_NTK 
      spectrum_scaled = spectrum_scaled * jnp.heaviside(spectrum_scaled-1e-16, 0)
      eigs_L = spectrum_scaled / norms
      normed_spec[i,:] = eigs_L

    spectrum_true = normed_spec / np.outer(len(layers), degens)

    # scale coeffs Ak defined so that Q_k(x*x') = Ak \sum_m Y_{km}(x) Y_{km}(x')
    # A_k^2  N(k,d) = < Q_k(z)^2 >
    Q_scaled = Q_lax * w[jnp.newaxis,:]
    Ak_sqr = jnp.einsum('ij,ij->i', Q_lax, Q_scaled) / degens

    return spectrum_true, Ak_sqr



