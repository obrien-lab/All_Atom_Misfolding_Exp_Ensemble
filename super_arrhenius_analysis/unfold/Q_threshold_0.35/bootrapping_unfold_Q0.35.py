"""
Script for fitting disentanglement time at 298K using data at 550-800K.
No restriction of a parameter in curver fit.
"""
import numpy as np
from scipy.optimize import curve_fit

"""
Unfold time where Q<=Q_threshold for 1ns
"""
Q_threshold = 0.35
FPT_550 = np.array(
    [6.34, 11.01, 2.85, 7.45, 11.58, 9.14, 4.54, 10.43, 8.7, 7.01, 5.07, 5.98, 14.82, 8.64, 12.86, 4.54, 7.31, 6.71,
     7.69, 10.59, 10.12, 8.42, 4.9, 3.37, 6.9, 8.68, 11.34, 20.37, 6.25, 11.49, 6.33, 4.64, 2.65, 5.99, 7.56, 7.69,
     11.21, 5.77, 5.69, 3.33, 6.82, 6.62, 6.38, 24.89, 9.03, 4.47, 13.6, 9.15, 11.21, 5.86])
FPT_600 = np.array(
    [1.74, 1.6, 2.95, 1.76, 1.08, 1.86, 2.16, 5.42, 2.02, 1.34, 2.12, 2.73, 1.57, 2.77, 2.54, 2.46, 2.71, 3.18, 4.73,
     1.75, 4.95, 2.78, 2.97, 1.32, 3.87, 2.1, 3.6, 1.7, 1.82, 4.14, 1.31, 3.33, 0.43, 2.06, 2.82, 1.69, 0.96, 1.5, 4.24,
     0.77, 1.8, 1.9, 2.22, 5.29, 1.81, 1.09, 1.8, 3.14, 1.46, 1.98])
FPT_650 = np.array(
    [1.39, 0.6, 0.97, 0.41, 0.66, 0.82, 0.53, 1.04, 0.92, 0.94, 0.68, 1.66, 0.65, 1.14, 0.54, 0.74, 1.03, 0.68, 1.16,
     1.44, 0.8, 0.9, 1.03, 1.67, 1.04, 1.34, 0.61, 0.65, 0.58, 0.95, 0.82, 0.79, 0.7, 0.54, 1.53, 0.47, 0.62, 1.39, 0.6,
     1.23, 0.48, 1.07, 1.88, 0.26, 0.77, 1.08, 0.84, 1.03, 0.68, 0.64])
FPT_700 = np.array(
    [0.52, 0.4, 0.43, 0.3, 0.68, 0.49, 0.38, 0.33, 0.4, 0.24, 0.59, 0.37, 0.38, 0.4, 0.23, 0.27, 0.65, 0.3, 0.6, 0.22,
     0.48, 0.62, 0.38, 0.33, 0.49, 0.46, 0.58, 0.39, 0.29, 0.32, 0.2, 0.37, 0.4, 0.41, 0.42, 0.25, 0.2, 0.28, 0.59,
     0.21, 0.42, 0.4, 0.58, 0.44, 0.4, 0.23, 0.55, 0.31, 0.27, 0.27])
FPT_750 = np.array(
    [0.41, 0.28, 0.15, 0.18, 0.36, 0.29, 0.3, 0.18, 0.22, 0.31, 0.34, 0.22, 0.4, 0.23, 0.34, 0.19, 0.24, 0.25, 0.3,
     0.27, 0.29, 0.27, 0.2, 0.15, 0.21, 0.35, 0.21, 0.25, 0.38, 0.24, 0.31, 0.3, 0.18, 0.26, 0.15, 0.23, 0.24, 0.17,
     0.24, 0.3, 0.29, 0.29, 0.26, 0.25, 0.17, 0.22, 0.21, 0.27, 0.32, 0.27])
FPT_800 = np.array(
    [0.21, 0.13, 0.15, 0.15, 0.13, 0.11, 0.11, 0.12, 0.13, 0.08, 0.16, 0.18, 0.22, 0.24, 0.2, 0.17, 0.18, 0.15, 0.25,
     0.1, 0.14, 0.19, 0.23, 0.13, 0.14, 0.14, 0.18, 0.11, 0.13, 0.16, 0.16, 0.2, 0.12, 0.13, 0.24, 0.21, 0.21, 0.13,
     0.09, 0.16, 0.09, 0.21, 0.19, 0.15, 0.15, 0.15, 0.2, 0.22, 0.08, 0.25])
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
# print('1st row: ln(rate), 2nd row: rate [1/s], 3rd row: time [s]')
print("#1/T       ln(kapp)    2.5%ln(kapp)    97.5%ln(kapp)", file=f)
for i in range(len(data4fit[:, 0])):
    print(
        f"{data4fit[i, 0]:10.5f}    {data4fit[i, 1]:10.5f}    {np.percentile(data4fit[i, 2:], 2.5):10.5f}   {np.percentile(data4fit[i, 2:], 97.5):10.5f}",
        file=f)
print('#########################################################')

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
