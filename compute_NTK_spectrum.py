import numpy as np
import math
import matplotlib.pyplot as plt
import gegenbauer
import scipy as sp
import scipy.special
import scipy.optimize


# useful function for ReLU
def f(phi, L):
    if L==1:
        return np.arccos(1/math.pi * np.sin(phi) + (1 - 1/math.pi *  np.arccos(np.cos(phi))   )   * np.cos(phi))
    elif L==0:
        return np.arccos(np.cos(phi))
    else:
        return f(phi,L-1)

def NNGP(phi, L):
    z = np.cos(phi)
    for l in range(L):
        z = 1/np.pi * ( np.sqrt(1-z**2) + z * (np.pi - np.arccos(z)) )
    return z

def NNGP2(z,L):
    for l in range(L):
        z = 1/np.pi * ( np.sqrt(1-z**2) + z * (np.pi - np.arccos(z)) )
    return z

# recursively compute ReLU NNGP for depth L and angles phi
#def NNGP(phi, L):
#    if L == 1:

#        z = np.cos(phi)
#        k = (1-1/math.pi * phi) * z + 1/math.pi * np.sqrt(1-z**2)
#        return k
#    else:
#        Theta = NNGP(phi ,L-1)#
#        return f(Theta,1)

def Dot_Phi_Kernel(z):
    return 1/np.pi *(np.pi - np.arccos(z))

def NTK_iter(phi, L):
    all_NNGP = [NNGP(phi,l) for l in range(L)]
    all_dot = [Dot_Phi_Kernel(np.arccos(phi),)]
    return

# recursively compute ReLU NTK for depth L and angles phi
def NTK_recurse(phi, L):
    if L ==1:
        z = np.cos(phi)
        k = 2*(1-1/math.pi * phi) * z + 1/math.pi * np.sqrt(1-z**2)
        return k
    else:
        a = phi
        for i in range(L-1):
            a = f(a,1)
        return np.cos(f(a,1)) + NTK_recurse(phi,L-1) * (1-a/math.pi)

# simpler NTK recursion relying on NNGP computation
def NTK_recurse2(z, L):
    if L == 0:
        return z # first layer is just x * x' = z
    else:
        dot_Phi_L = Dot_Phi_Kernel( NNGP2(z,L-1) )
        Phi_last = NNGP2(z,L)
        return Phi_last + NTK_recurse2(z,L-1) * dot_Phi_L

# gegenbauer quadrature with 5000 roots to compute kernel decomposition
def get_effective_spectrum(layers, kmax, d, ker = 'NTK'):

    all_coeffs = np.zeros((len(layers), kmax))
    num_pts = 5000
    alpha = d/2.0 - 1
    z, w = sp.special.roots_gegenbauer(num_pts, alpha)
    degens = np.array( [gegenbauer.degeneracy(d,k) for k in range(kmax)] )
    Q = gegenbauer.get_gegenbauer_fast2(z, kmax, d)
    NTK_mat = np.zeros((len(layers), num_pts))
    for i in range(len(layers)):
        l = layers[i]
        if ker == 'NTK':
            NTK_mat[i,:] = NTK_recurse2(z, l)
        else:
            NTK_mat[i,:] = NNGP2(z, l)


    scaled_NTK = NTK_mat * np.outer( np.ones(len(layers)), w)
    scaled_Q =  Q * np.outer(degens, np.ones(num_pts))

    spectrum_scaled = scaled_NTK @ scaled_Q.T * gegenbauer.surface_area(d-1)/gegenbauer.surface_area(d)
    spectrum_scaled = spectrum_scaled * np.heaviside(spectrum_scaled-1e-14, 0)
    for i in range(kmax):
        if spectrum_scaled[0,i] < 1e-18:
            spectrum_scaled[0,i] = 0


    khat = Q.T @ spectrum_scaled[0,:]
    k = NTK_mat[0,:]

    spectrum_true = spectrum_scaled / np.outer(len(layers), degens)
    for i in range(len(layers)):
        for j in range(kmax-1):
            if spectrum_true[i,j+1] < spectrum_true[i,j]*1e-5:
                spectrum_true[i,j+1] = 0


    return spectrum_true


# use hermite quadrature to get NTK decomposition for higher input dimension d
def get_effective_spectrum_hermite(layers, kmax, d, ker = 'NTK'):

    all_coeffs = np.zeros((len(layers), kmax))
    num_pts = 2000
    alpha = d/2.0 - 1
    z, w = sp.special.roots_hermite(num_pts)
    degens = np.array( [gegenbauer.degeneracy(d,k) for k in range(kmax)] )
    #Q = gegenbauer.get_gegenbauer_fast2(z, kmax, d)

    scales = np.array( [2**k * math.factorial(k)*np.sqrt(math.pi) for k in range(kmax)] )

    inds_valid = [i for i in range(num_pts) if np.abs(z[i]) < np.sqrt(d)]
    z_valid = z[inds_valid]
    w_valid = w[inds_valid]
    num_pts = len(inds_valid)
    H = gegenbauer.get_hermite_fast(z_valid,kmax, d)
    NTK_mat = np.zeros((len(layers), num_pts))
    max_element = np.amax( np.abs(z_valid/np.sqrt(d)))
    for i in range(len(layers)):
        l = layers[i]
        if ker == 'NTK':
            NTK_mat[i,:] = NTK_recurse2(np.arccos(z_valid/np.sqrt(d)), l)
        else:
            NTK_mat[i,:] = NNGP2(np.arccos(z_valid / np.sqrt(d)), l)

    scaled_NTK = NTK_mat * np.outer( np.ones(len(layers)), w_valid)
    scaled_H =  H * np.outer(scales**(-1), np.ones(num_pts))
    spectrum_scaled = scaled_NTK @ scaled_H.T

    spectrum_scaled = spectrum_scaled * np.heaviside(spectrum_scaled - 1e-30*np.ones(len(spectrum_scaled)), 0)

    spectrum_true = np.zeros(spectrum_scaled.shape)

    for i in range(len(layers)):
        spectrum_true[i,:] = gegenbauer.hermite_to_gegenbauer_coeffs(spectrum_scaled[i,:], d)

    for i in range(len(layers)):
        for j in range(kmax-1):
            if spectrum_true[i,j+1] < spectrum_true[i,j]*1e-5:
                spectrum_true[i,j+1] = 0

    return spectrum_true


def one_layer_NNGP(kmax, d):
    num_pts = 10000
    alpha= d/2.0 -1
    z,w=  sp.special.roots_gegenbauer(num_pts, alpha)
    NNGP = (1/math.pi * (1-z**2)**(0.5) + (1-1/math.pi * np.arccos(z)) * z)
    Q = gegenbauer.get_gegenbauer(z, kmax, d)

    norms = np.array([gegenbauer.normalizing_factor(k,d/2.0-1) for k in range(kmax)])
    degens = np.array( [gegenbauer.degeneracy(d,k) for k in range(kmax)] )

    NNGP_coeffs = Q @ (w*NNGP) / (norms*degens)

    for i in range(kmax):
        if NNGP_coeffs[i] < 1e-25:
            NNGP_coeffs[i] = 0
    return NNGP_coeffs
