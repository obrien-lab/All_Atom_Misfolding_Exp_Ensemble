"""
Script for fitting disentanglement time at 298K using data at 550-800K.
No restriction of a parameter in curver fit.
"""
import numpy as np
from scipy.optimize import curve_fit

"""
Disentanglement time where G0+G1=0 for 1ns
Data for 2ww4-disentanglement
"""
G_threshold = 0
FPT_550 = np.array(
    [43.378, 24.58, 55.008, 20.67, 58.528, 43.906, 39.756, 36.67, 33.082, 16.892, 33.052, 48.464, 35.226, 74.906,
     24.002, 25.268, 27.104, 24.062, 37.918, 18.946, 26.492, 34.086, 32.242, 46.512, 41.472, 17.336, 15.596, 20.208,
     37.198, 33.698])
FPT_600 = np.array(
    [8.926, 6.914, 9.006, 14.12, 7.302, 14.846, 15.586, 8.95, 10.816, 7.512, 11.69, 16.386, 13.268, 17.604, 17.266,
     14.416, 12.562, 9.388, 6.696, 5.946, 10.758, 12.18, 9.91, 7.152, 21.418, 14.296, 11.306, 12.938, 16.122, 8.518])
FPT_650 = np.array(
    [4.192, 5.988, 6.124, 2.894, 18.226, 4.198, 1.814, 5.982, 12.666, 4.938, 8.184, 5.298, 10.866, 5.002, 3.584, 6.696,
     5.318, 6.068, 5.23, 4.338, 14.442, 3.478, 6.182, 5.966, 2.022, 3.77, 7.75, 8.43, 4.664, 5.928])
FPT_700 = np.array(
    [4.22, 3.014, 2.498, 2.94, 2.718, 7.296, 2.154, 3.008, 5.32, 2.314, 2.286, 3.942, 4.248, 2.872, 8.212, 3.948, 9.212,
     7.576, 5.914, 5.1, 10.142, 2.954, 7.312, 3.35, 3.254, 1.852, 4.496, 3.394, 2.95, 5.304])
FPT_750 = np.array(
    [1.438, 2.308, 2.494, 3.21, 1.87, 1.084, 3.032, 1.838, 1.412, 2.694, 1.778, 1.478, 1.52, 2.89, 3.542, 1.086, 1.374,
     1.678, 1.614, 1.27, 2.374, 1.172, 2.73, 3.604, 1.598, 1.988, 2.592, 3.094, 1.204, 1.418])
FPT_800 = np.array(
    [5.002, 2.036, 2.71, 1.522, 5.002, 5.002, 2.682, 0.926, 2.808, 1.628, 1.638, 5.002, 1.98, 5.002, 3.736, 5.002,
     1.556, 2.116, 2.432, 3.342, 1.9, 2.004, 5.002, 2.432, 2.532, 2.846, 1.208, 5.002, 0.984, 2.878])
"""
Variable control
"""
nboots = 10000  # number of bootstrap resampling iterations
dt = 0.1  # time interval in ns used to calculate survival probability
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
    """
    Write survival probability for plotting
    """
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
    ln_kapp = quadratic_fun(1 / target_temp, *kopt)
    data4fit[-1, i + 2] = ln_kapp
    rateTable[-1, i + 2] = np.exp(data4fit[-1, i + 2])
    timeTable[-1, i + 2] = 1 / rateTable[-1, i + 2]

print('#########################################################')
print('Write datapoint file, this file is used for plot super-arrhenius analysis')

f = open("datapoints.dat", "w")
print('#Write datapoint file, this file is used for plot super-arrhenius analysis', file=f)

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

print("Disentanglement time (in ns) with CI95%")
real_time = time_array[0]
time_lower = np.percentile(time_array[1:], 2.5)
time_upper = np.percentile(time_array[1:], 97.5)

print(real_time, time_lower, time_upper)

print(f"Disentanglement time (in s) at {target_temp} K, scaled with 143 to get real time, with CI95%")
real_time_scaled = real_time * 143 / (10 ** 9)
time_lower_scaled = time_lower * 143 / (10 ** 9)
time_upper_scaled = time_upper * 143 / (10 ** 9)
print(f"{real_time_scaled:.3f} s, (95% CI: [{time_lower_scaled:.3f}, {time_upper_scaled:.3f}])")

print('Write to datapoints file...')
print(f"#Disentanglement time (in s) at {target_temp} K, scaled with 143 to get real time, with CI95%", file=f)
print(f"#G <= {G_threshold} for 1ns", file=f)

print(f"#{real_time_scaled:.3f} s, (95% CI: [{time_lower_scaled:.3f}, {time_upper_scaled:.3f}])", file=f)

f.close()

## plot with error bars
T_range = np.arange(0.0012, 0.0035, 0.00001)
# write datafile:
fx_fit = quadratic_fun(T_range, *kopt_target)
datafitting = np.array([T_range, fx_fit]).T
np.savetxt('function.dat', datafitting, delimiter='\t', fmt='%10.5f')
