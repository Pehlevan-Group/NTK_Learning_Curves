import numpy as np
import scipy as sp
import scipy.special
import scipy.misc
import math
import matplotlib.pyplot as plt
import numba
from numba import jit


def get_gegenbauer_fast(x,kmax, d):
    alpha = d/2 - 1
    Q = np.zeros((kmax, len(x)))
    Q[0,:] = 1
    Q[1,:] = 2*alpha * x
    for k_m in range(kmax - 2):
        k = k_m + 2
        Q[k,:] = 1/k * (2 * x*Q[k-1,:]*(k + alpha -1) - (k+2*alpha - 2) * Q[k-2,:] )

    return Q

def get_gegenbauer_fast2(x,kmax,d):
    dim_inp = len(x.shape)
    if dim_inp == 1:
        Q = np.zeros((kmax, len(x)))
        Q[0,:] = np.ones(len(x))
        Q[1,:] = x
        for km in range(kmax-2):
            k = km + 1
            Q[k+1,:] = (2*k+d-2)/(k+d-2) * x * Q[k,:] - k/(k+d-2) * Q[k-1,:]
    else:
        Q = np.zeros((kmax, x.shape[0], x.shape[1]))
        Q[0,:,:] = np.ones((x.shape[0], x.shape[1]))
        Q[1,:,:] = x
        for km in range(kmax-2):
            k = km + 1
            Q[k+1,:,:] = (2*k+d-2)/(k+d-2) * x * Q[k,:,:] - k/(k+d-2) * Q[k-1,:,:]
    return Q

def get_gegenbauer(x, kmax, d):


    alpha = d/2 - 1

    Q = np.zeros((kmax, len(x)))
    for k in range(kmax):
        Q[k,:] = sp.special.eval_gegenbauer(k,alpha, x)
    return Q

def normalizing_factor(k,alpha):
    return math.pi * 2**(1-2*alpha) * sp.special.gamma(k+2*alpha) / (math.factorial(k) * (k+alpha) * sp.special.gamma(alpha)**2)

def check_orthogonality(num_pts, kmax,d):
    x = np.linspace(-1,1,num=num_pts)
    Q = get_gegenbauer(x, kmax, d)
    alpha = d/2.0 - 1
    weight = (1-x**2)**(alpha-1.0/2)
    all_prods = np.zeros((kmax, kmax))
    for k in range(kmax):
        for kp in range(kmax):
            Q_k = Q[k,:]
            Q_kp = Q[kp,:]
            prod = np.dot(Q_k*weight, Q_kp) * 2/num_pts
            all_prods[k,kp] = prod / normalizing_factor(k, alpha)
    return all_prods

def degeneracy(d,k):

    if d>100:
        if k ==0:
            return 1
        else:
            return (2*k+d-2)/(k * math.factorial(k-1)) * (d-1)**(k-1)
    else:
        if k==0:
            return 1
        else:
            return (2*k+d-2)/k * sp.special.comb(k+d-3,k-1)

def surface_area(d):
    return 2*math.pi**(d/2) / sp.special.gamma(d/2)

def surface_area_ratio(d):
    return np.sqrt(math.pi) * (d/2.0-0.5)**(-0.5)

# assume f is defined on uniform support over [-1,1]
def inner_product(f, z, Q, d):
    num_pt = len(f)
    alpha = d/2.0 - 1
    kmax = Q.shape[0]
    weight = (1-z**2)**(alpha - 0.5)
    coeffs = np.zeros(kmax)
    for k in range(kmax):
        coeffs[k] = np.dot(Q[k,:], weight*f) / normalizing_factor(k,alpha)
    return coeffs

def get_hermite(x, kmax, d):
    H = np.zeros((kmax, len(x)))
    for k in range(kmax):
        H[k,:] = sp.special.eval_hermite(k, x)
    return H

# physicists hermite polynomials
def get_hermite_fast(x, kmax, d):
    H = np.zeros((kmax, len(x)))
    H[0,:] = np.ones(len(x))
    H[1,:] = 2*x
    for k in range(kmax- 2):
        kp = k+2
        H[kp,:] = 2*x*H[kp-1,:] - 2*(kp-1) * H[kp-2,:]
    return H

def hermite_to_gegenbauer_coeffs(coeffs, d):
    kmax = len(coeffs)
    wd = surface_area(d)
    wdm = surface_area(d-1)
    surface_ratio = surface_area_ratio(d)
    degens = np.array( [degeneracy(d,k) for k in range(kmax)] )
    rescales = np.sqrt(2) * np.array([1/surface_ratio *np.sqrt(2*math.pi/d)* math.factorial(k) * 2**k  / degens[k] for k in range(kmax)])**(0.5)
    return coeffs * rescales

def hermite_to_gegenbauer_activation_coeffs(kmax, d, nonlinearity = 'relu'):

    num_pts = 50*kmax
    x, w = sp.special.roots_hermite(num_pts) # physicists Hermite polynomials
    z = np.maximum(x, np.zeros(len(x)))
    #H = get_hermite(x, kmax, d)
    H = get_hermite_fast(x,kmax, d)
    scales = [2**k * math.factorial(k)*np.sqrt(math.pi) for k in range(kmax)]

    coeffs = (H @ ( (z/np.sqrt(d)) * w)) / scales
    reconstruction = H.T @ coeffs

    coeffs_gegenbauer = calculate_activation_coeffs(kmax, d)
    coeffs2 = hermite_to_gegenbauer_coeffs(coeffs, d)

    coeffs2 = coeffs2 * np.heaviside(coeffs2**2 - 1e-30 * np.ones(len(coeffs2)),0)

    return coeffs2


def calculate_activation_coeffs(kmax,d, nonlinearity = 'relu'):

    if d > 100:
        return hermite_to_gegenbauer_activation_coeffs(kmax, d, nonlinearity)
    num_pts = 50*kmax
    alpha = d/2.0-1
    x, w = sp.special.roots_gegenbauer(num_pts, alpha)
    Q = get_gegenbauer_fast2(x, kmax, d)

    #norms = np.array([normalizing_factor(k,alpha) for k in range(kmax)])
    degens = np.array( [degeneracy(d,k) for k in range(kmax)] )
    if nonlinearity == 'tanh':
        z = np.tanh(x)
    else:
        z = np.maximum(x, np.zeros(len(x)))

    coeffs = Q @ (z * w) * surface_area(d-1)/surface_area(d)
    coeffs = coeffs * np.heaviside(coeffs**2 - 1e-30 * np.ones(len(coeffs)),0)

    return coeffs


def monte_carlo_coeffs(kmax, d, nonlinearity = 'relu'):

    num_pts = 1000
    variance = 1/(d-3)
    z = np.random.normal(0, np.sqrt(variance), num_pts)

    Q = get_gegenbauer(z, kmax, d)
    if nonlinearity == 'tanh':
        sig = np.tanh(z)
    else:
        sig = np.maximum(z, np.zeros(len(z)))

    norm_factors = np.array(  [ normalizing_factor(k,d/2.0-0.5) for k in range(kmax)] )
    coeffs = 1/num_pts * (Q @ ( sig ) ) * (norm_factors)**(-1)

    reconstruction = Q.T @ coeffs
    plt.scatter(z, sig)
    plt.scatter(z, reconstruction)
    plt.show()
    return coeffs
