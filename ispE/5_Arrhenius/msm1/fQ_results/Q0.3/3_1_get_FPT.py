import numpy as np
import argparse

def find_first_consecutive_fQ_below_threshold(data, threshold=0.2, consecutive_points=100):
    """
    Find the first time where `fQ` is less than or equal to the specified threshold for the given number of consecutive time points.

    Parameters:
    data (list of tuples): A list of (time, fQ) tuples.
    threshold (float): The threshold value for `fQ`.
    consecutive_points (int): The number of consecutive points required to meet the condition.

    Returns:
    float: The first time where `fQ` is below the threshold for the specified number of consecutive points, or -1 if not found.
    """
    count = 0  # To count consecutive points below threshold

    for i in range(len(data)):
        if data[i][1] <= threshold:
            count += 1
            if count == consecutive_points:
                return data[i - consecutive_points + 1][0]
        else:
            count = 0  # Reset count if the condition is not met

    return -1  # Return -1 if the condition is never met


def cal_Su_list(first_time_zero, time_end, dt=0.01):
    """
    Function to calculate the survival probability
    """
    survival_probability = np.empty((0, 2), float)
    time_range = np.arange(0, time_end, dt)
    for time_interval in time_range:
        prob = np.count_nonzero(first_time_zero >= time_interval) / len(first_time_zero)
        survival_probability = np.append(survival_probability, np.array([[time_interval, prob]]), axis=0)

    return survival_probability[:, 0], survival_probability[:, 1]


def write_SU(FPT, Temp):
    time_end = np.max(FPT) + 5
    ts, Su_list = cal_Su_list(FPT, time_end)
    with open(f'survival_{Temp}.dat', 'w') as f:
        for t, s in zip(ts, Su_list):
            f.write(f'{t:.2f}    {s:.3f}\n')


def main():
    parser = argparse.ArgumentParser(description="Process temperature data for survival probability.")
    parser.add_argument('-T', '--temperature', type=int, default=None,
                        help='Specify the temperature in Kelvin. If not specified, run for all default temperatures.')
    args = parser.parse_args()

    temperatures = [600, 650, 700, 750, 800]
    if args.temperature:
        temperatures = [args.temperature]

    for temperature in temperatures:
        FPT_list = []  # unit of ns
        for i in range(1, 51):
            data = np.loadtxt(f'../{temperature}/fQ_{i}.dat', delimiter=',')
            FPT = find_first_consecutive_fQ_below_threshold(data, threshold=0.3) / 1000
            FPT_list.append(FPT)
            # print(f"{i}: {FPT}")

        print(f"Max SU time for {temperature} K: {np.max(FPT_list)}")
        print(', '.join(map(str, FPT_list)))
        write_SU(FPT_list, temperature)


if __name__ == "__main__":
    main()

