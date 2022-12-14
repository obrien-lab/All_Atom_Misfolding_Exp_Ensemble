"""
Script for fitting disentanglement time at 298K using data at 550-800K.
No restriction of a parameter in curver fit.
"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt

"""
Unfold time where Q<= Q_threshold for 1ns
"""
Q_threshold = 0.3
FPT_550 = np.array(
    [9.73, 13.96, 5.85, 9.57, 14.57, 11.13, 5.34, 17.97, 11.22, 7.73, 6.84, 6.83, 15.34, 8.99, 13.6, 5.6, 9.65, 7.13,
     9.19, 11.46, 11.61, 8.91, 5.19, 4.46, 11.84, 9.3, 11.6, 24.41, 6.67, 14.11, 6.71, 5.03, 3.35, 6.94, 8.4, 16.19,
     21.64, 9.09, 6.69, 4.55, 7.47, 9.76, 9.78, 27.26, 10.67, 4.66, 17.76, 10.77, 14.42, 8.22])
FPT_600 = np.array(
    [2.15, 1.71, 4.1, 1.89, 1.36, 1.91, 2.52, 6.08, 3.35, 1.6, 2.24, 3.27, 2.14, 3.03, 2.86, 2.9, 2.97, 3.46, 5.41,
     2.64, 7.33, 3.75, 3.29, 1.43, 4.08, 2.66, 3.81, 2.45, 2.13, 5.28, 1.38, 3.53, 0.97, 2.9, 5.27, 1.9, 1.03, 1.84,
     4.57, 1.73, 2.41, 2.53, 2.83, 5.8, 2.09, 2.52, 1.9, 3.64, 1.8, 2.07])
FPT_650 = np.array(
    [1.89, 1.09, 1.18, 0.88, 0.76, 1.0, 0.65, 1.43, 1.16, 1.45, 0.82, 1.83, 0.82, 2.27, 0.55, 0.83, 1.86, 0.99, 1.28,
     1.95, 1.15, 1.3, 1.26, 1.67, 1.41, 1.46, 0.66, 0.91, 1.19, 1.25, 1.02, 0.89, 0.72, 0.78, 1.82, 0.6, 0.66, 1.63,
     0.79, 1.27, 0.76, 1.29, 2.43, 0.4, 0.81, 1.32, 1.04, 1.29, 0.81, 0.74])
FPT_700 = np.array(
    [0.65, 0.49, 0.53, 0.38, 0.89, 0.56, 0.41, 0.38, 0.5, 0.3, 0.8, 0.46, 0.55, 0.46, 0.29, 0.36, 0.72, 0.36, 0.61,
     0.27, 0.58, 0.81, 0.48, 0.34, 0.97, 0.5, 0.8, 0.49, 0.54, 0.41, 0.23, 0.48, 0.45, 0.46, 0.49, 0.31, 0.27, 0.54,
     0.75, 0.22, 0.45, 0.46, 0.77, 0.59, 0.5, 0.43, 0.64, 0.4, 0.38, 0.3])
FPT_750 = np.array(
    [0.42, 0.33, 0.16, 0.28, 0.44, 0.33, 0.35, 0.21, 0.5, 0.46, 0.4, 0.33, 0.45, 0.28, 0.41, 0.23, 0.28, 0.29, 0.43,
     0.27, 0.4, 0.31, 0.3, 0.23, 0.22, 0.42, 0.22, 0.34, 0.46, 0.26, 0.34, 0.38, 0.22, 0.34, 0.18, 0.26, 0.28, 0.24,
     0.35, 0.42, 0.33, 0.4, 0.28, 0.31, 0.21, 0.25, 0.24, 0.34, 0.36, 0.3])
FPT_800 = np.array(
    [0.23, 0.16, 0.19, 0.27, 0.16, 0.13, 0.2, 0.15, 0.18, 0.1, 0.17, 0.19, 0.23, 0.28, 0.23, 0.21, 0.26, 0.23, 0.29,
     0.12, 0.16, 0.24, 0.26, 0.14, 0.16, 0.21, 0.21, 0.13, 0.15, 0.2, 0.2, 0.23, 0.14, 0.19, 0.33, 0.24, 0.26, 0.15,
     0.12, 0.18, 0.17, 0.22, 0.21, 0.18, 0.17, 0.16, 0.24, 0.24, 0.13, 0.29])
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

# write datafile:
fx_fit = quadratic_fun(T_range, *kopt_target)
datafitting = np.array([T_range, fx_fit]).T
np.savetxt('function.dat', datafitting, delimiter='\t', fmt='%10.5f')
