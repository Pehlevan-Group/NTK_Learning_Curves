import numpy as np
import scipy as sp


def f(z, *args):
    p, spectrum, degens, lamb = args
    return z - lamb - z*np.sum( degens*spectrum/(p*spectrum+z) )

def fp(z, *args):
    p, spectrum, degens, lamb = args
    return 1 - np.sum( degens * spectrum/(p*spectrum+z) ) + z*np.sum(degens * spectrum/(p*spectrum + z)**2)

def solve_implicit(pvals, spectrum, degens, lamb):

    zvals = np.zeros(len(pvals))
    for i,p in enumerate(pvals):
        args = (p, spectrum, degens, lamb)
        zvals[i] = sp.optimize.root_scalar(f = f, method = 'newton', fprime = fp, x0=2*np.sum(degens*spectrum)+2*lamb,args = args).root
    return zvals


def gamma_fn(pvals, zvals, spectrum,degens):
    all_vals = np.zeros(len(pvals))
    for i, p in enumerate(pvals):
        all_vals[i] = p * np.sum(degens * spectrum**2/(spectrum * p + zvals[i])**2 )
    return all_vals


# computes theoretical expressions for each spectral learning curve
def learning_curves_modes(pvals, spectrum, degens, lamb):
    zvals = solve_implicit(pvals, spectrum, degens, lamb) # t + lambda
    gamma = gamma_fn(pvals, zvals, spectrum, degens)
    errs = np.zeros((len(spectrum),len(pvals)))
    for j in range(len(spectrum)):
        lamb_j = spectrum[j]
        errs[j] = zvals**2/(1-gamma) * 1.0/(lamb_j*pvals+zvals)**2
    return errs



