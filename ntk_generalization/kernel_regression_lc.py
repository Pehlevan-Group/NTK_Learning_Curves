import numpy as np
import scipy as sp
import scipy.special
import scipy.stats
import gegenbauer
import compute_NTK_spectrum
import matplotlib.pyplot as plt
import approx_learning_curves
#import compute_NTK_spectrum
import pandas as pd
import argparse

def sample_random_points(num_pts, d):
    R = np.random.multivariate_normal(np.zeros(d), np.eye(d), num_pts)
    for i in range(num_pts):
        R[i,:] = R[i,:] / np.linalg.norm(R[i,:])
    return R

def compute_kernel(X, Xp, spectrum, d, kmax):
    P = X.shape[0]
    Pp = Xp.shape[0]
    gram = X @ Xp.T
    gram = np.reshape(gram, P*Pp)
    Q = gegenbauer.get_gegenbauer_fast2(gram, kmax, d)
    degens = np.array( [gegenbauer.degeneracy(d,k) for k in range(kmax)] )
    K = Q.T @ (spectrum * degens)
    K = np.reshape(K, (P,Pp))
    return K

def generalization(P, X_teach, spectrum, kmax, d, num_repeats, lamb = 1e-6):

    errors_avg = np.zeros(kmax)
    errors_tot_MC = 0
    all_errs = np.zeros((kmax, num_repeats))
    all_MC = np.zeros(num_repeats)
    X_teach = sample_random_points(P_teach,d)
    alpha_teach = np.sign( np.random.random_sample(P_teach) - 0.5* np.ones(P_teach) )

    for i in range(num_repeats):

        X_teach = sample_random_points(P_teach,d)
        alpha_teach = np.sign( np.random.random_sample(P_teach) - 0.5* np.ones(P_teach) )

        X = sample_random_points(P, d)
        K_student = compute_kernel(X,X, spectrum, d, kmax)
        K_stu_te = compute_kernel(X,X_teach, spectrum, d, kmax)
        y = K_stu_te @ alpha_teach


        K_inv = np.linalg.inv(K_student + lamb * np.eye(P))
        alpha = K_inv @ y

        degens = np.array( [gegenbauer.degeneracy(d,k) for k in range(kmax)] )

        gram_ss = X @ X.T
        gram_st = X @ X_teach.T
        gram_tt = X_teach @ X_teach.T


        Q_ss = gegenbauer.get_gegenbauer_fast2(np.reshape(gram_ss, P**2), kmax, d)
        Q_st = gegenbauer.get_gegenbauer_fast2(np.reshape(gram_st, P*P_teach), kmax, d)
        Q_tt = gegenbauer.get_gegenbauer_fast2(np.reshape(gram_tt, P_teach**2), kmax, d)

        errors = np.zeros(kmax)
        for k in range(kmax):
            Q_ssk = np.reshape(Q_ss[k,:], (P,P))
            Q_stk = np.reshape(Q_st[k,:], (P,P_teach))
            Q_ttk = np.reshape(Q_tt[k,:], (P_teach,P_teach))
            errors[k] = spectrum[k]**2 * degens[k] * ( alpha.T @ Q_ssk @ alpha - 2*alpha.T @ Q_stk @ alpha_teach + alpha_teach.T @ Q_ttk @ alpha_teach )
        errors_avg += 1/num_repeats * errors
        all_errs[:,i] = errors

        num_test = 2500
        X_test = sample_random_points(num_test, d)
        K_s = compute_kernel(X,X_test, spectrum, d, kmax)
        K_t = compute_kernel(X_teach,X_test, spectrum, d, kmax)

        y_s = K_s.T @ alpha
        y_t = K_t.T @ alpha_teach
        tot_error = 1/num_test * np.linalg.norm(y_s - y_t)**2
        print("errors")
        print("expt:   %e" % tot_error)
        print("theory: %e" % np.sum(errors))

        errors_tot_MC += 1/num_repeats *  tot_error
        all_MC[i] = tot_error

    std_errs = sp.stats.sem(all_errs, axis=1)
    std_MC = sp.stats.sem(all_MC)

    return errors_avg, errors_tot_MC, std_errs, std_MC


parser = argparse.ArgumentParser()
parser.add_argument('--input_dim', type=int, default= 10,
  help='data input dimension')
parser.add_argument('--lamb', type=float,
  help='explicit regularization penalty', default = 0)
parser.add_argument('--NTK_depth', type=int, default= 3,
  help='depth of Fully Connected ReLU NTK')

args = parser.parse_args()
d = args.input_dim
lamb = args.lamb
depth = args.NTK_depth


kmax = 30
degens = np.array( [gegenbauer.degeneracy(d,k) for k in range(kmax)] )
spectrum = compute_NTK_spectrum.get_effective_spectrum([depth], kmax, d, ker = 'NTK')[0,:]

s = [i for i in spectrum if i > 0]
P = 50
P_teach = 300
P_vals = np.logspace(0.25, 3, num = 15).astype('int')
num_repeats = 50

all_errs = np.zeros((len(P_vals), kmax))
all_mc = np.zeros(len(P_vals))
std_errs = np.zeros( (len(P_vals), kmax) )
std_MC = np.zeros(len(P_vals))
for i in range(len(P_vals)):
    P = P_vals[i]
    all_errs[i,:], all_mc[i], std_errs[i,:], std_MC[i] = generalization(P, P, spectrum, kmax, d, num_repeats, lamb=lamb)


sol, p = approx_learning_curves.simulate_uc(spectrum, degens, lamb = lamb)

plt.rcParams.update({'font.size': 12})

kplot = [0,1,2,4,6]
colors = ['b','r','g', 'm', 'c']

all_errsdf = pd.DataFrame(all_errs)
std_errsdf = pd.DataFrame(std_errs)
mc_df = pd.DataFrame(all_mc)
std_mc_df = pd.DataFrame(std_MC)
all_errsdf.to_csv('results/all_errs_lamb%lf_d%d.csv' % (lamb, d))
std_errsdf.to_csv('results/std_errs_lamb%lf_d%d.csv' % (lamb, d))
mc_df.to_csv('results/mc_errs_lamb%lf_d%d.csv' % (lamb, d))
std_mc_df.to_csv('results/std_mc_errs_lamb%lf_d%d.csv' % (lamb, d))


for i in range(len(kplot)):
    plt.errorbar(P_vals, np.log10(all_errs[:,kplot[i]]), std_errs[:,kplot[i]] / all_errs[:,kplot[i]], marker = 'o',linestyle='none', color = colors[i], label = 'k=%d' % kplot[i])
    plt.plot(p, np.log10(sol[:,kplot[i]] * P_teach), color = colors[i])

plt.xscale('log')
plt.xlim([np.amin(p)+5, np.amax(P_vals)])
plt.legend()
plt.xlabel(r'p', fontsize=20)
plt.ylabel(r'$\log \ E_k$', fontsize=20)
plt.tight_layout()
plt.savefig('results/kernel_expt_learning_curve_lamb%lf_d%d.pdf' % (lamb, d))
plt.show()


plt.errorbar(P_vals, np.log10(all_mc), std_MC/all_mc, fmt = 'o', linestyle = 'none', label = 'expt test')
plt.plot(p, np.log10(np.sum(sol, axis=1) * P_teach), label= 'continuous approximation')
plt.xscale('log')
plt.xlim([np.amin(P_vals), np.amax(P_vals)])
plt.legend()
plt.xlabel(r'p')
plt.ylabel(r'$\log \ E_g$')
plt.savefig('results/sup_mode_vs_test_lamb%lf_d%d.pdf' % (lamb,d))
plt.show()
