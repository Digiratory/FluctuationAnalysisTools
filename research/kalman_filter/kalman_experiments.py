import multiprocessing

import numpy as np
import pandas as pd

from StatTools.experimental.analysis.tools import get_extra_h_dfa
from StatTools.experimental.augmentation.perturbations import add_poisson_gaps
from StatTools.filters.kalman_filter import FractalKalmanFilter
from StatTools.generators.kasdin_generator import create_kasdin_generator


def process_single_iter(args):
    """Обрабатывает одну итерацию с заданным H, s (порядок ФФ) и возвращает результаты"""
    h, s, r_list, trj_len, n_times = args
    results_local = []
    print(f"H={h}")
    for _ in range(n_times):
        signal = get_signal(h, trj_len, s)
        for r in r_list:
            gaps_signals = {}
            gaps_signals["gaps_short_signal"] = add_poisson_gaps(signal, 0.1, s / 2)
            gaps_signals["gaps_eq_signal"] = add_poisson_gaps(signal, 0.1, s)
            gaps_signals["gaps_long_signal"] = add_poisson_gaps(signal, 0.1, s * 2)
            for name, gap_signal in gaps_signals.items():
                restored_signal = apply_kalman_filter(signal, gap_signal, h, r)
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


def get_r_list() -> list:
    return [2, 4, 8]


def get_signal(h: float, length: int, s: int, normalize=False) -> np.array:
    """Get normalized signal."""
    generator = create_kasdin_generator(
        h, length=length, filter_coefficients_length=s, normalize=normalize
    )
    signal = generator.get_full_sequence()
    if not normalize:
        return signal


def apply_kalman_filter(
    orig_signal: np.array, signal, model_h: np.array, r: int, noise=None
) -> np.array:
    f = FractalKalmanFilter(dim_x=r, dim_z=1)
    if noise is None:
        noise = np.zeros(len(orig_signal))
    f.set_parameters(model_h, np.std(noise) ** 2, kasdin_lenght=len(signal), order=r)
    estimated_signal = np.zeros(len(signal))
    for k in range(1, len(signal)):
        f.predict()
        if not np.isnan(signal[k]):
            f.update(signal[k])
        estimated_signal[k] = f.x[0].item()
    return estimated_signal


def get_metrics(orig_signal, restored_signal) -> dict:
    h_restored = get_extra_h_dfa(restored_signal)
    h_orig = get_extra_h_dfa(orig_signal)
    mse_restored = np.nanmean((orig_signal - restored_signal) ** 2)
    return {"H_restored": h_restored, "MSE": mse_restored, "H_signal": h_orig}


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
    file_name = "kalman-beta.csv"
    metrics_df.to_csv(file_name, index=False)
    print(f"Matrics saved to {file_name}")
