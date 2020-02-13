import numpy as np
import scipy as sp
import scipy.optimize
import scipy.integrate
import scipy.special
import matplotlib.pyplot as plt
import compute_NTK_spectrum as cNTK
import gegenbauer
import math

def uc_implicit(x, *args):
    spectrum, p, lamb = args
    return 1 - np.sum( spectrum / (x * np.ones(len(spectrum)) + p * spectrum) )

def fprime(x, *args):
    spectrum, p, lamb = args
    return np.sum( spectrum/(x*np.ones(len(spectrum)))**(2) )

def upper_cont_approx(spectrum, p, lamb = 1):

    solver = sp.optimize.root_scalar(uc_implicit, x0=p, fprime = fprime, method='newton', args = (spectrum, p, lamb))
    f = solver.root
    return f

# solve for t = Tr <G(p)>
def implicit_total_err_eq(t, *args):
    p, spectrum, degens, lamb = args
    denom = (lamb+t)*np.ones(len(spectrum)) + p*spectrum
    f = t - (lamb + t) * np.sum(spectrum * degens /  denom)
    return f

# solve for t(p) = Tr<G(p)>
def solve_total(pvals, spectrum, degens, lamb):
    roots = np.zeros(len(pvals))
    for i in range(len(pvals)):
        p = pvals[i]
        args = (p,spectrum, degens, lamb)
        sol = sp.optimize.root_scalar(implicit_total_err_eq, x0=0.01*np.sum(spectrum**2*degens), x1=500*np.sum(spectrum**2*degens), args=args)
        root = sol.root
        conv = sol.converged
        roots[i] = root
    return roots

# gamma as defined in Proposition 3 of the main text
def gamma(p,t, spectrum, degens, lamb):

    numerator = degens * spectrum**2 * (lamb + t)**2
    denom = ( (lamb + t)*np.ones(len(spectrum)) + spectrum*p)**2
    return np.sum(numerator/denom)

#  <G^2>
def total_err(p, t, spectrum, degens, lamb):
    gam = gamma(p,t,spectrum, degens,lamb)
    return (lamb + t)**2 * gam / ( (lamb+t)**2 - p*gam)

# get mode error
def mode_err_MOC(p,t,spectrum, degens, lamb):
    mode_errs = np.zeros(len(spectrum))
    prefactor = (1 + p/(lamb+t)**2 * total_err(p,t,spectrum, degens, lamb))
    for i in range(len(spectrum)):
        mode_errs[i] = prefactor * spectrum[i]**2 * degens[i] / (1 + spectrum[i]*p/(lamb+t) )**2
    return mode_errs


def dynamical(g, p, *args):
    degens, lamb = args
    return - degens**(-1) * g**2/(lamb + np.sum(g))


def dynamics_errors(e, p, *args):
    degens, lamb = args
    return -2 * e**(1.5) * (degens)**(-1) /(lamb + np.dot( degens**(0.5), e**(0.5) ))

# calculate theoretical learning curves using the Algorithm 1 of the main text
def simulate_uc(spectrum, degens, lamb = 1e-8, num_pts = 500, max_p= 3.5):

    p = np.logspace(1e-1, max_p, num = num_pts)

    total_roots = solve_total(p, spectrum, degens, lamb)

    errs = np.zeros(len(p))
    gams = np.zeros(len(p))
    mode_errs = np.zeros(( len(p), len(spectrum) ))
    for i in range(len(p)):
        errs[i] = total_err(p[i], total_roots[i], spectrum,degens,lamb)
        gams[i] = gamma(p[i],total_roots[i],spectrum, degens, lamb)
        mode_errs[i,:] = mode_err_MOC(p[i], total_roots[i], spectrum, degens, lamb)


    theory_err_nn = np.zeros(mode_errs.shape)
    for k in range(mode_errs.shape[1]):
        if spectrum[k] !=0:
            theory_err_nn[:,k] = mode_errs[:,k] / spectrum[k]

    kplot = [0,1,2,4,6]
    colors = ['b','r','g','m','c']
    return mode_errs, p
