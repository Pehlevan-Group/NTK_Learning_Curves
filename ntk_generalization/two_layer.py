import numpy as np
from ntk_generalization.utils import gegenbauer, compute_NTK_spectrum
import matplotlib.pyplot as plt
from ntk_generalization import approx_learning_curves
import csv
import numba
from numba import jit
from numba import prange
import time
import pandas as pd
import argparse

def SGD(X, Y, Theta, r, num_iter, readout_only=False):
    P = X.shape[0]
    d = X.shape[1]
    M = Theta.shape[0]

    deltaTheta = Theta.copy()
    batch_size = 40
    r = np.zeros(M)

    m_r = np.zeros(M)
    v_r = np.zeros(M)
    beta_1 = 0.9
    beta_2 = 0.999
    m_Theta = np.zeros((M,d))
    v_Theta = np.zeros((M,d))

    for t in range(num_iter):
        #r_grad = np.zeros(M)
        #Theta_grad = np.zeros((M,d))
        r_grad = np.zeros((batch_size, M))
        Theta_grad = np.zeros((batch_size, M, d))
        if t % 500==0:
            print("SGD epoch = %d" % t)
            Z0 = Theta @ X.T
            Z = np.maximum(Z0, np.zeros(Z0.shape))
            E_tr = 1/P * np.linalg.norm(Z.T @ r - Y)**2
            print("Etr = %e" % E_tr)
            if E_tr < 1e-16:
                break

        g = np.zeros(M)
        g_Theta = np.zeros((M,d))

        # batch wise computation
        inds = np.random.randint(0,P, batch_size)
        x_t = X[inds, :]
        y_t = Y[inds]
        Z = Theta @ x_t.T
        A = np.maximum(Z, np.zeros((M, batch_size)))
        Deriv = np.heaviside(Z, np.zeros((M, batch_size)))
        f_t = A.T @ r # batchsize
        g = A @ (f_t - y_t)
        r_deriv = np.outer(r, np.ones(batch_size)) * Deriv
        f_y_x = x_t * np.outer(f_t - y_t, np.ones(d))
        g_Theta = r_deriv @ f_y_x


        m_r = beta_1*m_r + (1-beta_1)*g
        v_r = beta_2*v_r + (1-beta_2)*(g**2)
        m_hat = m_r / (1-beta_1)
        v_hat = v_r / (1-beta_2)

        m_Theta = beta_1 * m_Theta + (1-beta_1) * g_Theta
        v_Theta = beta_2 * v_Theta + (1-beta_2) * g_Theta**2
        m_Theta_hat = m_Theta / (1-beta_1)
        v_Theta_hat = v_Theta / (1-beta_2)

        delta_r = - 1e-3 / d * m_hat / (np.sqrt(v_hat) + 1e-8*np.ones(M))
        delta_Theta = - 1e-3 /d * m_Theta_hat / (np.sqrt(v_Theta_hat) + 1e-8 *np.ones((M,d)))

        r += delta_r
        Theta += delta_Theta


    Z0 = Theta @ X.T
    Z = np.maximum(Z0, np.zeros(Z0.shape))
    E_tr = 1/P * np.linalg.norm(Z.T @ r - Y)**2

    return Theta, r, E_tr

def sample_random_points(num_pts, d):
    R = np.random.multivariate_normal(np.zeros(d), np.eye(d), num_pts)

    R = R* np.outer( np.linalg.norm(R, axis=1)**(-1), np.ones(d) )
    return R

@jit(nopython=True, parallel=True)
def sample_random_points_jit(num_pts, d, R):
    for i in prange(R.shape[0]):
        for j in prange(R.shape[1]):
            R[i,j] = np.random.standard_normal()

    for i in prange(R.shape[0]):
        R[i,:] = R[i,:] * (np.linalg.norm(R[i,:]) + 1e-10)**(-1)
    return R

@jit(nopython = True)
def feedfoward(X, Theta, r):
    Z0 = Theta @ X.T
    Z = np.maximum(Z0, np.zeros(Z0.shape))
    return Z.T @ r

def compute_kernel(X, Xp, spectrum, d, kmax):
    P = X.shape[0]
    Pp = Xp.shape[0]
    gram = X @ Xp.T
    gram = np.reshape(gram, P*Pp)
    #Q = gegenbauer.get_gegenbauer(gram, kmax, d)
    Q = gegenbauer.get_gegenbauer_fast2(gram, kmax, d)
    degens = np.array( [gegenbauer.degeneracy(d,k) for k in range(kmax)] )
    K = Q.T @ (spectrum * degens)
    #K = Q.T @ spectrum
    K = np.reshape(K, (P,Pp))
    return K

def get_gegenbauer_gram(Theta1, Theta2):
    gram = Theta1 @ Theta2.T
    M = Theta1.shape[0]
    perc = 0
    Q = gegenbauer.get_gegenbauer_fast2(np.reshape(gram, M**2), kmax, d)

    return Q

#@jit(nopython=True)
def get_mode_errs(Theta, Theta_teach, r, r_teach, kmax, d, degens):

    M = Theta.shape[0]
    Q_ss = get_gegenbauer_gram(Theta, Theta)
    Q_st = get_gegenbauer_gram(Theta, Theta_teach)
    Q_tt = get_gegenbauer_gram(Theta_teach, Theta_teach)

    mode_errs=np.zeros(kmax)
    for k in range(kmax):
        Q_ssk = np.reshape(Q_ss[k,:], (M,M))
        Q_stk = np.reshape(Q_st[k,:], (M,M))
        Q_ttk = np.reshape(Q_tt[k,:], (M,M))
        mode_errs[k] = spectrum[k] * degens[k] * ( r.T @ Q_ssk @ r - 2*r.T @ Q_stk @ r_teach + r_teach.T @ Q_ttk @ r_teach )

    return mode_errs

#@jit(nopython=True, parallel=True)
def generalization_expt(P, spectrum, M, d, kmax, num_repeats, Theta_teach, r_teach, num_test=1000):


    all_mode_errs = np.zeros((num_repeats, kmax))
    all_mc_errs = np.zeros(num_repeats)
    all_training_errs = np.zeros(num_repeats)
    degens = np.array( [gegenbauer.degeneracy(d,k) for k in range(kmax)] )
    print("P = %d" % P)
    Theta = np.zeros((M,d))
    #Theta_teach = np.zeros((M,d))
    X_test = np.zeros((num_test, d))
    X = np.zeros((P, d))
    for t in range(num_repeats):
        print("t=%d" %t)
        start = time.time()

        Theta = sample_random_points_jit(M, d, Theta)
        r = np.random.standard_normal(M) / np.sqrt(M)

        X = sample_random_points_jit(P, d, X)
        Z_teach = np.maximum(Theta_teach @ X.T, np.zeros((M,P)) )
        Y = Z_teach.T @ r_teach

        num_iter = min(200*P, 40000)
        Theta, r, E_tr = SGD(X, Y, Theta, r, num_iter, readout_only=False)
        print("final Etr = %e" % E_tr)
        counter = 1

        print("finished SGD")
        print("num tries: %d" % counter)

        all_mode_errs[t,:] = get_mode_errs(Theta,Theta_teach, r, r_teach, kmax, d, degens)
        end = time.time()
        print("time = %lf" %(end -start))

        X_test = sample_random_points_jit(num_test, d, X_test)
        #X_test = sample_random_points(num_test, d)
        Y_test = feedfoward(X_test, Theta_teach, r_teach)
        Y_pred = feedfoward(X_test, Theta, r)
        all_mc_errs[t] = 1/num_test * np.linalg.norm(Y_test-Y_pred)**2
        all_training_errs[t] = E_tr
    average_mode_errs = np.mean(all_mode_errs, axis = 0)
    std_errs = np.std(all_mode_errs, axis=0)
    average_mc =np.mean(all_mc_errs)
    std_mc = np.std(all_mc_errs)
    print("average MC   = %e" % average_mc)
    print("sum of modes = %e" % np.sum(average_mode_errs))
    return average_mc, std_mc, np.mean(all_training_errs)

def compute_kernel(X, Xp, spectrum, d, kmax):
    P = X.shape[0]
    Pp = Xp.shape[0]
    gram = X @ Xp.T
    gram = np.reshape(gram, P*Pp)
    #Q = gegenbauer.get_gegenbauer(gram, kmax, d)
    Q = gegenbauer.get_gegenbauer_fast2(gram, kmax, d)
    degens = np.array( [gegenbauer.degeneracy(d,k) for k in range(kmax)] )
    K = Q.T @ (spectrum * degens)
    #K = Q.T @ spectrum
    K = np.reshape(K, (P,Pp))
    return K

def generalization_expt_kteach(P, spectrum, M, d, kmax, num_repeats, X_teach, alpha_teach, spectrum_teach, num_test=1000):

    all_mode_errs = np.zeros((num_repeats, kmax))
    all_mc_errs = np.zeros(num_repeats)
    all_training_errs = np.zeros(num_repeats)
    degens = np.array( [gegenbauer.degeneracy(d,k) for k in range(kmax)] )
    print("P = %d" % P)
    Theta = np.zeros((M,d))
    X_test = np.zeros((num_test, d))
    X = np.zeros((P, d))
    for t in range(num_repeats):
        print("t=%d" %t)
        start = time.time()

        Theta = sample_random_points_jit(M, d, Theta)
        r = np.random.standard_normal(M) / np.sqrt(M)

        X = sample_random_points_jit(P, d, X)

        K = compute_kernel(X_teach, X)
        Y = K.T @ alpha_teach


        num_iter = 3*P
        Theta, r, E_tr = SGD(X, Y, Theta, r, num_iter, readout_only=False)
        print("Etr = %e" % E_tr)
        counter = 1

        print("finished SGD")
        print("num tries: %d" % counter)

        all_mode_errs[t,:] = get_mode_errs(Theta,Theta_teach, r, r_teach, kmax, d, degens)
        end = time.time()
        print("time = %lf" %(end -start))

        X_test = sample_random_points_jit(num_test, d, X_test)
        #X_test = sample_random_points(num_test, d)
        Y_test = feedfoward(X_test, Theta_teach, r_teach)
        Y_pred = feedfoward(X_test, Theta, r)
        all_mc_errs[t] = 1/num_test * np.linalg.norm(Y_test-Y_pred)**2
        all_training_errs[t] = E_tr
    average_mode_errs = np.mean(all_mode_errs, axis = 0)
    std_errs = np.std(all_mode_errs, axis=0)
    average_mc =np.mean(all_mc_errs)
    std_mc = np.std(all_mc_errs)
    print("average MC   = %e" % average_mc)
    print("sum of modes = %e" % np.sum(average_mode_errs))
    return average_mc, std_mc, np.mean(all_training_errs)



parser = argparse.ArgumentParser()
parser.add_argument('--input_dim', type=int, default= 30,
  help='data input dimension')
parser.add_argument('--M', type=int,
  help='number of hidden units', default = 500)

args = parser.parse_args()
d = args.input_dim
M = args.M

kmax = 25
P_vals = [10,20,50,100,250,500]
num_repeats = 10


# calculate spectrum of teacher
spectrum = gegenbauer.calculate_activation_coeffs(kmax, d)**2
degens = np.array( [gegenbauer.degeneracy(d,k) for k in range(kmax)] )


# fix get effective spectrum for higher d
theory_spectrum = compute_NTK_spectrum.get_effective_spectrum([1], kmax, d, ker = 'NTK')[0,:]
theory_spectrum_hermite = compute_NTK_spectrum.get_effective_spectrum_hermite([2], kmax, d, ker='NTK')[0,:]
theory_spectrum_NNGP  = compute_NTK_spectrum.get_effective_spectrum([1], kmax, d, ker = 'NNGP')[0,:]

theory_g_sqr, p = approx_learning_curves.simulate_uc(theory_spectrum, degens, lamb = 1e-10)
theory_g_sqr_NNGP, p = approx_learning_curves.simulate_uc(theory_spectrum_NNGP, degens, lamb = 1e-10)
theory_g_sqr_hermite, p = approx_learning_curves.simulate_uc(theory_spectrum_hermite, degens, lamb = 1e-8)
theory_gen = np.zeros(theory_g_sqr.shape)
theory_gen_NNGP = np.zeros(theory_g_sqr.shape)
theory_gen_hermite = np.zeros(theory_g_sqr.shape)

for k in range(kmax):
    if spectrum[k] !=0:
        theory_gen[:,k] = theory_g_sqr[:,k] / theory_spectrum[k]**2 * spectrum[k]
        theory_gen_NNGP[:,k] = theory_g_sqr_NNGP[:,k] / theory_spectrum_NNGP[k]**2 * spectrum[k]
        theory_gen_hermite[:,k] = theory_g_sqr_hermite[:,k] / theory_spectrum[k]**2 * spectrum[k]
        #theory_gen[:,k] = theory_g_sqr[:,k] / spectrum[k] * M


colors = ['b','r','g', 'm', 'c']
kplot = [0,1,2,4,6]



mc_errs = np.zeros(len(P_vals))
std_mc_errs = np.zeros(len(P_vals))
training_errs = np.zeros(len(P_vals))
Theta_teach = sample_random_points(M, d)
r_teach = np.random.standard_normal(M) / np.sqrt(M)
for i in range(len(P_vals)):
    P = P_vals[i]
    av_mc, std_mc, E_tr = generalization_expt(P, spectrum, M, d, kmax,  num_repeats, Theta_teach, r_teach)
    mc_errs[i] = av_mc
    std_mc_errs[i] = std_mc
    training_errs[i] = E_tr

plt.rcParams.update({'font.size': 12})

plt.loglog(P_vals, training_errs)
plt.xlabel('P')
plt.ylabel(r'$E_{tr}$')
plt.savefig('train_errs.pdf')
plt.show()


colors = ['b','r','g', 'm', 'c']

mode_df = pd.DataFrame(mode_errs)
std_df = pd.DataFrame(std_errs)
training_df = pd.DataFrame(training_errs)
mc_df = pd.DataFrame(mc_errs)
std_mc_df = pd.DataFrame(std_mc_errs)

mode_df.to_csv('results/mode_errs_twolayer_M%d_d%d.csv' % (M,d))
std_df.to_csv('results/std_errs_twolayer_M%d_d%d.csv' % (M,d))
training_df.to_csv('results/train_errs_twolayer_M%d_d%d.csv' % (M,d))
mc_df.to_csv('results/mc_errs_twolayer_M%d_d%d.csv' % (M,d))
std_mc_df.to_csv('results/mc_std_twolayer_M%d%d.csv' % (M,d))



plt.errorbar(P_vals, np.log10(mc_errs), std_mc_errs / mc_errs, marker = 'o', label = 'expt test')
plt.errorbar(P_vals, np.log10(np.sum(mode_errs, axis=0)), np.sqrt(np.sum(std_errs[kplot[i],:]**2)) / np.sum(mode_errs, axis=0), marker = 'o', label = 'sum mode errors')
plt.plot(p, np.log10(np.sum(theory_gen, axis = 1)) , label = 'random matrix theory')
plt.xscale('log')
plt.legend()
plt.xlim([np.amin(p), 3*np.amax(P_vals)])
plt.xlabel(r'$P$')
plt.ylabel(r'$E_g$')
plt.savefig('results/total_err_two_layer_NTK_M_%d_d_%d.pdf' % (M,d))
plt.show()
