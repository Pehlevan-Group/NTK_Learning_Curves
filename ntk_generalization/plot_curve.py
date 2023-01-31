import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import approx_learning_curves
import compute_NTK_spectrum
import gegenbauer


# write file key here to read all of the data from the experiment
key = 'rough_4_layer_d30_M500_target10000_numiter10000_lr0'
file_name = 'results/all_test_losses_' + key
std_name = 'results/std_test_losses_' + key
train_name = 'results/train_loss_4_layer_30_M500_target10000_numiter10000_lr0'
kernel_name = 'results/all_kernel_test_' + key
kernel_std_name = 'results/all_kernel_std_test_' + key
x = pd.read_csv(file_name + '.csv')
print(x)
mode_errs = x.to_numpy()

x = pd.read_csv(std_name+'.csv')
std_errs = x.to_numpy()
print(x)


x = pd.read_csv(train_name + '.csv')
train_errs = x.to_numpy()
print("training errrors")
print(train_errs)

kernel_errs = pd.read_csv(kernel_name + '.csv').to_numpy()
kernel_std = pd.read_csv(kernel_std_name + '.csv').to_numpy()
print(mode_errs.shape)

pvals = [5,10,20,50,100,250,500,1000]
kmax = 20
d = 30
depth = 4

coeffs = compute_NTK_spectrum.get_effective_spectrum([depth - 1], kmax, d, ker = 'NTK')[0,:]

degens = np.array( [gegenbauer.degeneracy(d,k) for k in range(kmax)] )

theory_curves, p = approx_learning_curves.simulate_uc(coeffs, degens, lamb = 0, num_pts = 2000, max_p= np.log10(np.amax(pvals)))

plt.rcParams.update({'font.size': 12})

kvals = [1,2,4]
print(train_errs.shape)
for i in range(train_errs.shape[0]):
    plt.semilogy(train_errs[i,1:train_errs.shape[1]], label = 'k=%d' % (kvals[i]))
plt.legend()
plt.xlabel('SGD iteration', fontsize=20)
plt.ylabel(r'$E_{tr}$', fontsize=20)
plt.tight_layout()
plt.savefig(train_name + '.pdf')
plt.show()



ind = np.amin([i for i in range(len(p)) if p[i] > 5])
ind2 = np.amin([i for i in range(len(p)) if p[i] > 250])
ind50 = np.amin([i for i in range(len(p)) if p[i] > 50])
ind250 = np.amin([i for i in range(len(p)) if p[i] > 250])
colors = ['r','g','c']

kvals = [1,2,4]
for i in range(len(kvals)):
    k = kvals[i]
    if k != 0:
        k = kvals[i]
        theory_rescale = kernel_errs[i,1] / theory_curves[ind,k]
        plt.plot(p,  np.log10( (theory_curves[:,k]) * theory_rescale ), color = colors[i])
        plt.errorbar(pvals[0:8], np.log10(mode_errs[i,1:9] ), std_errs[i,1:9]/mode_errs[i,1:9] , linestyle='none', marker = 'o', color = colors[i], label = 'NN k = %d' % k)

for i in range(len(kvals)):
    k = kvals[i]
    if k != 0:
        k = kvals[i]
        plt.errorbar(pvals[0:8], np.log10(kernel_errs[i,1:9]), kernel_std[i,1:9]/kernel_errs[i,1:9], linestyle='none', marker = '^', color=colors[i], label = 'Kernel k=%d' % k)
plt.xscale('log')
plt.xlabel('p', fontsize = 20)
plt.ylabel(r'$\log \ E_k$', fontsize = 20)
plt.xlim([3,1250])
plt.legend()
plt.tight_layout()
plt.savefig(file_name + '.pdf')
plt.show()
