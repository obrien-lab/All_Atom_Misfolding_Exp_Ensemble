"""
Script for fitting disentanglement time at 298K using data at 550-800K.
No restriction of a parameter in curver fit.
"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt

"""
Unfolding time where Q<=Q_threshold for 1ns
"""
Q_threshold = 0.2
FPT_550 = np.array(
    [16.84, 24.98, 7.69, 11.24, 20.61, 17.98, 17.07, 46.56, 26.48, 12.67, 10.38, 9.02, 19.55, 11.41, 18.19, 7.04, 12.8,
     8.13, 13.12, 15.85, 13.63, 10.23, 13.51, 6.45, 28.15, 20.82, 25.86, 31.47, 11.29, 29.49, 15.85, 18.22, 4.93, 11.29,
     12.29, 17.67, 29.6, 16.88, 11.69, 5.85, 12.02, 14.17, 16.88, 29.22, 13.39, 6.22, 27.83, 12.22, 19.56, 12.33])
FPT_600 = np.array(
    [3.85, 3.63, 4.89, 2.68, 2.41, 3.11, 4.57, 12.09, 5.64, 2.44, 3.67, 5.1, 3.35, 4.95, 7.43, 5.85, 5.01, 6.7, 6.79,
     3.86, 11.26, 5.89, 5.61, 2.26, 5.37, 3.25, 5.03, 3.26, 2.69, 7.13, 1.85, 5.64, 2.03, 3.88, 7.34, 4.73, 2.31, 3.79,
     8.56, 2.43, 3.78, 3.0, 4.96, 7.82, 3.21, 3.79, 3.14, 6.38, 2.32, 2.75])
FPT_650 = np.array(
    [2.32, 1.76, 1.66, 1.35, 1.23, 1.5, 0.77, 1.84, 2.02, 1.8, 1.6, 2.55, 1.11, 3.17, 1.75, 0.98, 2.77, 1.48, 2.05,
     2.73, 1.5, 1.82, 1.95, 2.31, 1.74, 2.25, 1.57, 1.45, 1.3, 1.59, 1.42, 1.19, 1.19, 1.39, 2.78, 0.93, 1.13, 1.91,
     1.97, 1.73, 1.05, 1.65, 2.62, 1.0, 1.49, 1.73, 1.63, 1.86, 1.09, 1.52])
FPT_700 = np.array(
    [0.87, 0.61, 0.8, 0.64, 1.08, 0.95, 1.03, 0.66, 0.77, 0.49, 1.16, 0.6, 0.7, 0.54, 0.61, 0.45, 1.01, 0.63, 0.83,
     0.76, 0.62, 0.96, 0.79, 0.61, 1.06, 0.94, 0.95, 0.79, 0.82, 0.96, 0.46, 0.85, 0.69, 0.56, 0.84, 0.64, 0.43, 0.7,
     1.24, 0.35, 0.88, 0.56, 1.23, 0.76, 0.8, 0.73, 0.77, 0.78, 0.67, 0.59])
FPT_750 = np.array(
    [0.56, 0.47, 0.27, 0.34, 0.6, 0.37, 0.71, 0.4, 0.62, 0.52, 0.73, 0.57, 0.49, 0.32, 0.53, 0.41, 0.4, 0.33, 0.58,
     0.54, 0.48, 0.35, 0.39, 0.32, 0.3, 0.49, 0.3, 0.41, 0.55, 0.41, 0.43, 0.56, 0.38, 0.44, 0.39, 0.47, 0.48, 0.3,
     0.58, 0.71, 0.45, 0.46, 0.39, 0.63, 0.29, 0.44, 0.48, 0.42, 0.53, 0.34])
FPT_800 = np.array(
    [0.28, 0.31, 0.29, 0.48, 0.31, 0.22, 0.25, 0.28, 0.26, 0.21, 0.23, 0.33, 0.34, 0.43, 0.32, 0.34, 0.44, 0.31, 0.41,
     0.16, 0.33, 0.37, 0.34, 0.35, 0.29, 0.29, 0.27, 0.18, 0.18, 0.27, 0.34, 0.4, 0.21, 0.3, 0.47, 0.39, 0.31, 0.21,
     0.22, 0.26, 0.25, 0.26, 0.45, 0.21, 0.25, 0.24, 0.34, 0.39, 0.25, 0.4])
"""
Variable control
"""
nboots = 10000
dt = 0.1
target_temp = 298

if len(FPT_550) == len(FPT_600) == len(FPT_650) == len(FPT_700) == len(FPT_750) == len(FPT_800):
    traj_len = len(FPT_550)

else:
    print('number of trajectories at different temperatures are not equal...')
    exit

##############################################################

"""
Functions definition
"""


def cal_Su_list(first_time_zero, dt=0.01):
    """
    Function to calculate the survival probability
    """
    # time_end = 30
    time_end = np.max(first_time_zero) + 1.0
    survival_probability = np.empty((0, 2), float)
    time_range = np.arange(0, time_end, dt)
    for time_inteval in time_range:
        prob = np.count_nonzero(first_time_zero >= time_inteval) / len(first_time_zero)
        survival_probability = np.append(survival_probability, np.array([[time_inteval, prob]]), axis=0)

    return survival_probability[:, 0], survival_probability[:, 1]


def exp_fun(x, k, t0):
    """
    Function to fitting Su, get k and t0
    """
    t = x - t0
    t[t < 0] = 0
    return np.exp(-k * t)


def quadratic_fun(x, a, b, c):
    """
    x=1/T
    a,b,c fitting parameters
    """
    return a * x * x + b * x + c


def get_ln_kapp(FPT):
    """
    This function take first passage time array and return the ln of apparent rate
    """
    ts, Su_list = cal_Su_list(FPT)
    kopt, kcov = curve_fit(exp_fun, ts, Su_list, bounds=(0, np.inf), maxfev=5000)
    k, t0 = kopt[0], kopt[1]
    kapp = 1 / (t0 + 1 / k)
    return np.log(kapp)


def write_SU(FPT, Temp):
    ts, Su_list = cal_Su_list(FPT)
    with open(f'survival_{Temp}.dat', 'w') as f:
        for t, s in zip(ts, Su_list):
            f.write(f'{t:.2f}    {s:.3f}\n')


##############################################################

##############################################################
"""
First col is inverse temperature
Second row is real data, and following rows are bootstrapping data
This table contains the ln(rate), not rate
"""
data4fit = np.zeros((7, nboots + 2), dtype=float)
data4fit[:, 0] = [1 / 800, 1 / 750, 1 / 700, 1 / 650, 1 / 600, 1 / 550, 1 / target_temp]
rateTable = np.zeros((7, nboots + 2), dtype=float)
rateTable[:, 0] = [1 / 800, 1 / 750, 1 / 700, 1 / 650, 1 / 600, 1 / 550, 1 / target_temp]
timeTable = np.zeros((7, nboots + 2), dtype=float)
timeTable[:, 0] = [1 / 800, 1 / 750, 1 / 700, 1 / 650, 1 / 600, 1 / 550, 1 / target_temp]

"""
Fitting real data
"""
print("#########################\nreal data")
# 800K
data4fit[0, 1] = get_ln_kapp(FPT_800)
write_SU(FPT_800, 800)
rateTable[0, 1] = np.exp(data4fit[0, 1])
timeTable[0, 1] = 1 / rateTable[0, 1]
# 750K
data4fit[1, 1] = get_ln_kapp(FPT_750)
write_SU(FPT_750, 750)
rateTable[1, 1] = np.exp(data4fit[1, 1])
timeTable[1, 1] = 1 / rateTable[1, 1]
# 700K
data4fit[2, 1] = get_ln_kapp(FPT_700)
write_SU(FPT_700, 700)
rateTable[2, 1] = np.exp(data4fit[2, 1])
timeTable[2, 1] = 1 / rateTable[2, 1]

# 650K
data4fit[3, 1] = get_ln_kapp(FPT_650)
write_SU(FPT_650, 650)
rateTable[3, 1] = np.exp(data4fit[3, 1])
timeTable[3, 1] = 1 / rateTable[3, 1]

# 600K
data4fit[4, 1] = get_ln_kapp(FPT_600)
write_SU(FPT_600, 600)
rateTable[4, 1] = np.exp(data4fit[4, 1])
timeTable[4, 1] = 1 / rateTable[4, 1]

# 550K
data4fit[5, 1] = get_ln_kapp(FPT_550)
write_SU(FPT_550, 550)
rateTable[5, 1] = np.exp(data4fit[5, 1])
timeTable[5, 1] = 1 / rateTable[5, 1]

# fitting for 298K
T_inv = data4fit[:-1, 0]
lnKapp = data4fit[:-1, 1]
kopt_target, kcov = curve_fit(quadratic_fun, T_inv, lnKapp, maxfev=5000)
cc_target = np.corrcoef([lnKapp, quadratic_fun(T_inv, *kopt_target)])[0, 1] ** 2

# ln(kapp_298)
ln_kapp_target = quadratic_fun(1 / target_temp, *kopt_target)
print(f'T = {target_temp}, ln(kapp)= {ln_kapp_target}, R2= {cc_target}')

data4fit[-1, 1] = ln_kapp_target
rateTable[-1, 1] = np.exp(data4fit[-1, 1])
timeTable[-1, 1] = 1 / rateTable[-1, 1]
print(f'disentanglement time: {1 / np.exp(ln_kapp_target)} ns')

print("#########################\n")

for i in range(nboots):
    print(f'iteration: {i}')

    # 800K
    temp_dat = np.random.choice(FPT_800, traj_len, replace=True)
    data4fit[0, i + 2] = get_ln_kapp(temp_dat)
    rateTable[0, i + 2] = np.exp(data4fit[0, i + 2])
    timeTable[0, i + 2] = 1 / rateTable[0, i + 2]

    # 750K
    temp_dat = np.random.choice(FPT_750, traj_len, replace=True)
    data4fit[1, i + 2] = get_ln_kapp(temp_dat)
    rateTable[1, i + 2] = np.exp(data4fit[1, i + 2])
    timeTable[1, i + 2] = 1 / rateTable[1, i + 2]

    # 700K
    temp_dat = np.random.choice(FPT_700, traj_len, replace=True)
    data4fit[2, i + 2] = get_ln_kapp(temp_dat)
    rateTable[2, i + 2] = np.exp(data4fit[2, i + 2])
    timeTable[2, i + 2] = 1 / rateTable[2, i + 2]
    # 650K
    temp_dat = np.random.choice(FPT_650, traj_len, replace=True)
    data4fit[3, i + 2] = get_ln_kapp(temp_dat)
    rateTable[3, i + 2] = np.exp(data4fit[3, i + 2])
    timeTable[3, i + 2] = 1 / rateTable[3, i + 2]
    # 600K
    temp_dat = np.random.choice(FPT_600, traj_len, replace=True)
    data4fit[4, i + 2] = get_ln_kapp(temp_dat)
    rateTable[4, i + 2] = np.exp(data4fit[4, i + 2])
    timeTable[4, i + 2] = 1 / rateTable[4, i + 2]
    # 550K
    temp_dat = np.random.choice(FPT_550, traj_len, replace=True)
    data4fit[5, i + 2] = get_ln_kapp(temp_dat)
    rateTable[5, i + 2] = np.exp(data4fit[5, i + 2])
    timeTable[5, i + 2] = 1 / rateTable[5, i + 2]

    # fitting for Target temperature
    T_inv = data4fit[:-1, 0]
    lnKapp = data4fit[:-1, i + 2]
    kopt, kcov = curve_fit(quadratic_fun, T_inv, lnKapp, maxfev=5000)
    # ln(kapp_298)
    # ln_kapp = kopt[0] / (target_temp * target_temp) + kopt[1] / target_temp + kopt[2]
    ln_kapp = quadratic_fun(1 / target_temp, *kopt)
    data4fit[-1, i + 2] = ln_kapp
    rateTable[-1, i + 2] = np.exp(data4fit[-1, i + 2])
    timeTable[-1, i + 2] = 1 / rateTable[-1, i + 2]

print('#########################################################')
print('Write datapoint file, this file is used for plot arrhenius analysis')

f = open("datapoints.dat", "w")
print('#Write datapoint file, this file is used for plot arrhenius analysis', file=f)
print("#1/T       ln(kapp)    2.5%ln(kapp)    97.5%ln(kapp)", file=f)
for i in range(len(data4fit[:, 0])):
    print(
        f"{data4fit[i, 0]:10.5f}    {data4fit[i, 1]:10.5f}    {np.percentile(data4fit[i, 2:], 2.5):10.5f}   {np.percentile(data4fit[i, 2:], 97.5):10.5f}",
        file=f)

print('#########################################################')

""" 
Data to plot, this table contains mean values and +/- dX of a quantity, matplotlib accept this format for error plot.
Format of these table are:
[1/T]  [value]   [value-dx] [value+dx]  For plot
[1/T]  [value]   [lower] [upper]  For report
"""
data2plot = np.zeros((7, 4), dtype=float)
data2plot[:, 0] = [1 / 800, 1 / 750, 1 / 700, 1 / 650, 1 / 600, 1 / 550, 1 / target_temp]

# data to report with 95% CI, this contains lower and upper bound of quantity
data2report = np.zeros((7, 4), dtype=float)
data2report[:, 0] = [1 / 800, 1 / 750, 1 / 700, 1 / 650, 1 / 600, 1 / 550, 1 / target_temp]

for i in range(len(data4fit[:, 0])):
    data2plot[i, :] = data4fit[i, 0], data4fit[i, 1], data4fit[i, 1] - np.percentile(data4fit[i, 2:],
                                                                                     2.5), np.percentile(
        data4fit[i, 2:], 97.5) - data4fit[i, 1]

time_array = 1 / np.exp(data4fit[-1, 1:])  # 0 col is 1/T

print("unfolding time (in ns) with CI95%")
real_time = time_array[0]
time_lower = np.percentile(time_array[1:], 2.5)
time_upper = np.percentile(time_array[1:], 97.5)

print(real_time, time_lower, time_upper)

print(f"unfolding time (in s) at {target_temp} K, scaled with 143 to get real time, with CI95%")
real_time_scaled = real_time * 143 / (10 ** 9)
time_lower_scaled = time_lower * 143 / (10 ** 9)
time_upper_scaled = time_upper * 143 / (10 ** 9)
print(f"{real_time_scaled:.3f} s, (95% CI: [{time_lower_scaled:.3f}, {time_upper_scaled:.3f}])")

print('Write to datapoints file...')
print(f"#unfolding time (in s) at {target_temp} K, scaled with 143 to get real time, with CI95%", file=f)
print(f"#Q <= {Q_threshold} for 1ns", file=f)

print(f"#{real_time_scaled:.3f} s, (95% CI: [{time_lower_scaled:.3f}, {time_upper_scaled:.3f}])", file=f)

f.close()

T_range = np.arange(0.0012, 0.0035, 0.00001)
fx_fit = quadratic_fun(T_range, *kopt_target)
datafitting = np.array([T_range, fx_fit]).T
np.savetxt('function.dat', datafitting, delimiter='\t', fmt='%10.5f')
