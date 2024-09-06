#!/usr/bin/env python3

import sys, os, numpy as np
import mdtraj as mdt
import parmed as pmd
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt

######################################
matplotlib.rcParams['mathtext.fontset'] = 'stix'
#matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.labelsize'] = 'x-large'
matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['lines.markersize'] = 6
matplotlib.rcParams['xtick.major.width'] = 1
matplotlib.rcParams['ytick.major.width'] = 1
matplotlib.rcParams['xtick.labelsize'] = 'large'
matplotlib.rcParams['ytick.labelsize'] = 'large'
matplotlib.rcParams['legend.fontsize'] = 'large'

####################################
def exp_fun(x, k, t0):
    t = x - t0
    t[t<0] = 0
    return np.exp(-k*t)
    
def boot_fun(hit_time_list, ts):
    Su_list = [1]
    for i in range(len(ts)-1):
        idx_list = np.where(hit_time_list <= i)[0]
        Su_list.append(1-len(idx_list)/len(hit_time_list))
    kopt, kcov = curve_fit(exp_fun, ts, Su_list, bounds=(0, np.inf))
    return kopt
    
def bootstrap(boot_fun, data, n_time, *args):
    idx_list = np.arange(len(data))
    if len(data.shape) == 1:
        boot_stat = np.empty((n_time,), dtype=object)
    elif len(data.shape) == 2:
        boot_stat = np.empty((data.shape[0] ,n_time), dtype=object)
    else:
        print('bootstrap: Can only handle 1 or 2 dimentional data')
        sys.exit()
        
    for i in range(n_time):
        sample_idx_list = np.random.choice(idx_list, len(idx_list))
        if len(data.shape) == 1:
            new_data = data[sample_idx_list]
            boot_stat[i] = boot_fun(new_data, *args)
        elif len(data.shape) == 2:
            new_data = data[sample_idx_list, :]
            for j in range(data.shape[1]):
                boot_stat[i,j] = boot_fun(new_data[:,j], *args)
            
    return boot_stat
    
####################################

Temp= 750
data = np.loadtxt(f'survival_{Temp}.dat')
ts = data[:,0]
Su_list=data[:,1]
kopt, kcov = curve_fit(exp_fun, ts, Su_list, bounds=(0, np.inf))
cc = np.corrcoef([Su_list, exp_fun(ts,*kopt)])[0,1]**2
print('R2 for fitting: %.4f'%(cc))
print('k: %f; t0: %f'%(kopt[0], kopt[1]))


fig = plt.figure(figsize=(6,4))
plt.subplots_adjust(bottom=0.15, wspace=0.3, hspace=0.3)
ax = fig.add_subplot(1, 1, 1)
ax.plot(ts, Su_list, 'o', label='Data points')
ax.plot(ts, exp_fun(ts,*kopt), 
        label=('Fitting curve\n'+'$\\mathrm{R}^2$ = %.4f\nk = %f\nt0 = %f'%(cc, kopt[0], kopt[1])))
ax.set_xlabel('t (ns)')
ax.set_ylabel('Su')
ax.legend()

fig.savefig(f'Su_t0_{Temp}.png')
 

