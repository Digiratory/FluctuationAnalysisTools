import multiprocessing

import numpy as np
import pandas as pd

from StatTools.analysis.dfa import DFA
from StatTools.filters.kalman_filter import EnhancedKalmanFilter
from StatTools.generators.kasdin_generator import KasdinGenerator


def get_extra_h(signal):
    h = DFA(signal).find_h()
    new_h = h
    new_sihnal = signal
    if h > 1.5:
        # differentiate the signal
        diff_count = 0
        while new_h > 1.5:
            diff_count += 1
            new_sihnal = np.diff(new_sihnal)
            new_h = DFA(new_sihnal).find_h()
        new_h += diff_count

    elif h < 0.5:
        # integrate the signal
        integrate_count = 0
        while new_h < 0.5:
            integrate_count += 1
            new_sihnal = np.cumsum(new_sihnal)
            new_h = DFA(new_sihnal).find_h()
        new_h -= integrate_count
    return new_h


def process_single_iter(args):
    """Обрабатывает одну итерацию с заданным H, s (порядок ФФ) и возвращает результаты"""
    h, s, r_list, trj_len, n_times = args
    results_local = []
    print(f"H={h}")
    for _ in range(n_times):
        signal = get_signal(h, trj_len, s)
        for r in r_list:
            gaps_signals = {}
            gaps_signals["gaps_short_signal"] = get_gaps_signal(signal, s / 2)
            gaps_signals["gaps_eq_signal"] = get_gaps_signal(signal, s)
            gaps_signals["gaps_long_signal"] = get_gaps_signal(signal, s * 2)
            for name, gap_signal in gaps_signals.items():
                restored_signal = restore_signal(signal, gap_signal, r)
                metrics = get_metrics(signal, restored_signal)
                if name == "gaps_short_signal":
                    gaps_len = s / 2
                elif name == "gaps_eq_signal":
                    gaps_len = s
                else:
                    gaps_len = s * 2

                results_local.append(
                    {
                        "h": h,
                        "H_signal": metrics["H_signal"],
                        "H_restored": metrics["H_restored"],
                        "signal_length": len(signal),
                        "s": s,
                        "r": r,
                        "gaps_len": gaps_len,
                        "MSE": metrics["MSE"],
                    }
                )
    return results_local


def get_r_list() -> tuple:
    return list(range(1, 9))


def get_signal(h: float, length: int, s: int) -> np.array:
    """Get normalized signal."""
    generator = KasdinGenerator(
        h, length=length, filter_coefficients_length=s, normalize=True
    )
    signal = generator.get_full_sequence()
    # normalisation
    mean = np.nanmean(signal)
    std = np.nanstd(signal)
    return (signal - mean) / std - signal[0]


def get_gaps_signal(signal: np.array, gaps_parametr: float) -> np.array:
    return add_poisson_gaps(signal, 0.1, gaps_parametr)[0]


def restore_signal(orig_signal: np.array, signal: np.array, r: int) -> np.array:
    f = EnhancedKalmanFilter(dim_x=r, dim_z=1)
    f.auto_configure(orig_signal, np.zeros(len(orig_signal)))
    recovered_signal = np.zeros(len(signal))
    for k in range(1, len(signal)):
        f.predict()
        if not np.isnan(signal[k]):
            f.update(signal[k])
        recovered_signal[k] = f.x[0].item()
    return recovered_signal


def get_metrics(orig_signal, restored_signal) -> dict:
    h_restored = get_extra_h(restored_signal)
    h_orig = get_extra_h(orig_signal)
    mse_restored = np.nanmean((orig_signal - restored_signal) ** 2)
    return {"H_restored": h_restored, "MSE": mse_restored, "H_signal": h_orig}


def add_poisson_gaps(trajectory, gap_rate, length_rate):
    """
    Adds gaps to the trajectory according to the Poisson flow.

    Parameters:
    - trajectory: np.array, initial trajectory
    - gap_rate: parameter for the Poisson flow of gaps (the more, the more frequent the gaps)
    - length_rate: parameter for the Poisson distribution of gap lengths

    Returns:
    - trajectory_with_gaps: np.array, trajectory with gaps
    - gap_indices: list of tuples (start, end) of missed intervals
    """
    n = len(trajectory)
    trajectory_with_gaps = trajectory.copy()
    gap_indices = []
    current_pos = 0
    while current_pos < n:
        interval = np.random.exponential(1 / gap_rate)
        current_pos += int(interval)
        if current_pos >= n:
            break
        length = np.random.poisson(length_rate)
        if length <= 0:
            length = 1
        end_pos = min(current_pos + length, n)
        trajectory_with_gaps[current_pos:end_pos] = np.nan
        gap_indices.append((current_pos, end_pos))
        current_pos = end_pos
    return trajectory_with_gaps, gap_indices


def is_difference_less_or_equal(a, b, percent):
    """
    Проверяет, что разница между a и b не больше percent%

    Параметры:
    a, b: числа или массивы NumPy для сравнения
    percent: допустимая разница в процентах (0-100)

    Возвращает:
    bool или массив bool (если a и b - массивы)
    """
    relative_diff = np.abs(a - b) / np.maximum(np.abs(a), np.abs(b))
    return relative_diff <= (percent / 100)


def get_s_list(length: int) -> list:
    return list(range(1, 9)) + [length]


if __name__ == "__main__":
    H_LIST = np.arange(0.5, 5.25, 0.25)
    TRJ_LEN = 2**12
    n_times = 5
    print(H_LIST, TRJ_LEN)
    metrics_df = pd.DataFrame(
        columns=[
            "H_target",
            "H_signal",
            "H_restored",
            "signal_len",
            "s",
            "r",
            "gaps",
            "MSE",
        ]
    )
    print("Prepare args...")
    args_list = []
    r_list = get_r_list()
    s_list = get_s_list(TRJ_LEN)
    for h in H_LIST:
        for s in s_list:
            args_list.append((h, s, r_list, TRJ_LEN, n_times))
    print(f"Got {len(args_list) * len(r_list)} combinations for {n_times} times.")

    print("Run pool")
    with multiprocessing.Pool() as pool:
        results = pool.map(process_single_iter, args_list)

    print("Prepare results")
    for res in results:
        for row in res:
            metrics_df.loc[len(metrics_df)] = [
                row["h"],
                row["H_signal"],
                row["H_restored"],
                row["signal_length"],
                row["s"],
                row["r"],
                row["gaps_len"],
                row["MSE"],
            ]
    file_name = "kalman-beta-75.csv"
    metrics_df.to_csv(file_name, index=False)
    print(f"Matrics saved to {file_name}")
