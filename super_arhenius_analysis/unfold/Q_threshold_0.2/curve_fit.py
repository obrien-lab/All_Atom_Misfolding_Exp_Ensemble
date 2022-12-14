#!/usr/bin/env python3

import sys, os, numpy as np
import mdtraj as mdt
import parmed as pmd
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt

######################################
# config for matplotlib
font = {'family': 'arial',
        'weight': 'normal',
        'size': 14}

matplotlib.rc('font', **font)
# matplotlib.rcParams['axes.spines.right'] = False
# matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.linewidth'] = 2  # set the value globally

# set tick width
matplotlib.rcParams['xtick.major.size'] = 6
matplotlib.rcParams['xtick.major.width'] = 2
matplotlib.rcParams['ytick.major.size'] = 6
matplotlib.rcParams['ytick.major.width'] = 2


####################################
def exp_fun(x, k, t0):
    t = x - t0
    t[t < 0] = 0
    return np.exp(-k * t)


####################################

temperature_list = [550, 600, 650, 700, 750, 800]
for Temp in temperature_list:
    print(f'Temperature: {Temp}')
    data = np.loadtxt(f'survival_{Temp}.dat')
    ts = data[:, 0]
    Su_list = data[:, 1]
    kopt, kcov = curve_fit(exp_fun, ts, Su_list, bounds=(0, np.inf))
    cc = np.corrcoef([Su_list, exp_fun(ts, *kopt)])[0, 1] ** 2
    print('R2 for fitting: %.4f' % (cc))
    print('k: %f; t0: %f' % (kopt[0], kopt[1]))

    fig = plt.figure(figsize=(8, 4))
    plt.subplots_adjust(bottom=0.15, wspace=0.3, hspace=0.3)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ts, Su_list, 'o', markeredgewidth=1, label='Data points')
    ax.plot(ts, exp_fun(ts, *kopt), linewidth=4,
            label=('Fitting curve\n' + '$\\mathrm{R}^2$ = %.3f\nk = %.3f\n$\\mathrm{t_0}$ = %.3f' % (
            cc, kopt[0], kopt[1])))
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Su')
    ax.legend()
    plt.ylim([-0.05, 1.06])
    ax.set_yticks(np.arange(0, 1.1, step=0.5))
    fig.savefig(f'Su_t0_{Temp}.png', dpi=1000)
