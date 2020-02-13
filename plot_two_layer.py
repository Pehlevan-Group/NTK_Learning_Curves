import numpy as np
import gegenbauer
import compute_NTK_spectrum
import matplotlib.pyplot as plt
import approx_learning_curves
import csv
import numba
from numba import jit
from numba import prange
import time
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dim', type=int, default= 30,
  help='data input dimension')
parser.add_argument('--M', type=int,
  help='number of hidden units', default = 500)


args = parser.parse_args()
d = args.input_dim
M = args.M

kmax = 25
degens = np.array( [gegenbauer.degeneracy(d,k) for k in range(kmax)] )

theory_spectrum = compute_NTK_spectrum.get_effective_spectrum([1], kmax, d, ker = 'NTK')[0,:]


ak2 = gegenbauer.calculate_activation_coeffs(kmax,d)**2
bk2 = gegenbauer.calculate_activation_coeffs(kmax,d,nonlinearity='step')**2
spec_emp = np.zeros(kmax)
for i in range(kmax-1):
    if i==0:
        spec_emp[i] = ak2[i] + bk2[i+1] * 1/(d)
    elif i==1:
        spec_emp[i] = ak2[i] + bk2[0] + bk2[i+1] * (i+1)/(2*i+d) + (i+d-3)/(2*i+d-4) * bk2[i-1]
    else:
        spec_emp[i] = ak2[i] + bk2[i+1] * (i+1)/(2*i+d) + bk2[i-1] * (i+d-3)/(2*i+d-4)

print(spec_emp)
plt.loglog(spec_emp, 'o', label = 'sample on sphere')
plt.loglog(theory_spectrum, 'o', label = 'sample gaussian (NTK)')
plt.legend()
plt.xlabel('k')
plt.ylabel(r'$\lambda_k$')
plt.savefig('gauss_vs_sphere2layer.pdf')
plt.show()


theory_g_sqr, p = approx_learning_curves.simulate_uc(spec_emp, degens, lamb = 1e-10)
theory_NTK_g_sqr, p = approx_learning_curves.simulate_uc(theory_spectrum, degens, lamb = 1e-10)
theory_gen = np.zeros(theory_g_sqr.shape)
theory_NTK = np.zeros(theory_NTK_g_sqr.shape)
for k in range(kmax):
    if theory_spectrum[k] !=0:
        theory_NTK[:,k] = theory_NTK_g_sqr[:,k] / theory_spectrum[k]**2 * ak2[k]
    if spec_emp[k] != 0:
        theory_gen[:,k] = theory_g_sqr[:,k] / spec_emp[k]**2 * ak2[k]

colors = ['b','r','g', 'm', 'c']



colors = ['b','r','g', 'm', 'c']
kplot = [0,1,2,4,6]
P_vals = [10,20,50,100,250, 500]

mode_df = pd.read_csv('cluster/mode_errs_two_layer_M%d_d%d.csv' % (M,d))
std_df  = pd.read_csv('cluster/std_errs_two_layer_M%d_d%d.csv' % (M,d))
mc_df = pd.read_csv('cluster/mc_errs_two_layer_M%d_d%d.csv' % (M,d))
std_mc_df = pd.read_csv('cluster/std_mc_errs_two_layer_M%d_d%d.csv' % (M,d))

mode_errs = mode_df.to_numpy()
std_errs = std_df.to_numpy()
mc_errs = mc_df.to_numpy()
std_mc_errs = std_mc_df.to_numpy()

mode_errs = mode_errs[:,1:mode_errs.shape[0]]
std_errs = std_errs[:,1:mode_errs.shape[0]]
mc_errs = mc_errs[:,1:mode_errs.shape[0]]
std_mc_errs = std_mc_errs[:,1:mode_errs.shape[0]]

print(mode_errs)

plt.rcParams.update({'font.size': 12})

for i in range(len(kplot)):
    plt.errorbar(P_vals, np.log10(mode_errs[kplot[i],:]), std_errs[kplot[i],:] / mode_errs[kplot[i],:] , marker = 'o', color = colors[i], label = 'k=%d' % kplot[i])
    plt.plot(p, np.log10(2*theory_NTK[:,kplot[i]] ), '--', color = colors[i])

plt.xlim([np.amin(p), 3*np.amax(P_vals)])
plt.xscale('log')
plt.legend()
plt.xlabel(r'$P$', fontsize = 20)
plt.ylabel(r'$\log  E_g$', fontsize = 20)
plt.savefig('results/mode_errs_two_layer_NTK_M_%d_d_%d.pdf' % (M,d))
plt.show()


plt.errorbar(P_vals, np.log10(mc_errs), std_mc_errs / mc_errs, marker = 'o',linestyle='none', color = 'r', label = 'expt test error')
#plt.errorbar(P_vals, np.log10(np.sum(mode_errs, axis=0)), np.sqrt(np.sum(std_errs[kplot[i],:]**2)) / np.sum(mode_errs, axis=0), marker = 'o', label = 'sum mode errors')
plt.plot(p, np.log10(2*np.sum(theory_NTK, axis = 1)) , label = 'theory', color = 'r')
plt.xscale('log')
plt.legend()
plt.xlim([np.amin(p), 3*np.amax(P_vals)])
plt.xlabel(r'p', fontsize = 20)
plt.ylabel(r'$\log \ E_g$', fontsize = 20)
plt.tight_layout()
plt.savefig('results/total_err_two_layer_NTK_M_%d_d_%d.pdf' % (M,d))
plt.show()
