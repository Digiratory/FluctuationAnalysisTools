import multiprocessing
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from StatTools.experimental.analysis.tools import get_extra_h_dfa
from StatTools.experimental.augmentation.perturbations import (
    add_noise,
    add_poisson_gaps,
)
from StatTools.experimental.synthesis.tools import (
    adjust_hurst_to_range,
    reverse_hurst_adjustment,
)
from StatTools.filters.kalman_filter import FractalKalmanFilter
from StatTools.generators.kasdin_generator import create_kasdin_generator
from StatTools.utils import io

warnings.filterwarnings("ignore")

kalman_cache_folder = Path("/home/jovyan/git/FluctuationAnalysisTools/filter_matrices")


def process_snr_h_iter(args):
    """Обрабатывает одну итерацию с заданным H, s (порядок ФФ), SNR n_times и возвращает результаты"""
    h, s, r_list, trj_len, snr, n_times = args
    h_list = np.arange(0.5, 3.75, 0.25)

    results_local = []

    for _ in range(n_times):
        signal = get_signal(h, TRJ_LEN, s, normalize=False)
        signal -= np.mean(signal)
        signal /= np.std(signal)
        signal -= signal[0]
        h_s = get_extra_h_dfa(signal)
        adjusted_signal, applied_steps = adjust_hurst_to_range(signal)
        noisy_signal, noise = add_noise(adjusted_signal, ratio=snr)
        for r in r_list:
            for model_h in h_list:
                estimated_signal = apply_kalman_filter_cached(
                    noisy_signal,
                    model_h=model_h,
                    r=r,
                    noise=noise,
                    cache_folder=kalman_cache_folder,
                )
                estimated_signal = reverse_hurst_adjustment(
                    estimated_signal, applied_steps
                )
                se = np.nanstd(signal[0 : len(estimated_signal)] - estimated_signal)
                h_est = get_extra_h_dfa(estimated_signal)
                results_local.append(
                    {
                        "H_target": h,
                        "H_signal": h_s,
                        "H_estimated": h_est,
                        "signal_len": len(signal),
                        "s": s,
                        "r": r,
                        "SNR": snr,
                        "SE": se,
                        "H_kalman": model_h,
                    }
                )
    return results_local


def get_r_list() -> list:
    return np.array([2**i for i in range(1, 6)])


def get_signal(h: float, length: int, s: int, normalize=False) -> np.array:
    """Get normalized signal."""
    generator = create_kasdin_generator(
        h, length=length, filter_coefficients_length=s, normalize=normalize
    )
    return generator.get_full_sequence()


def apply_kalman_filter(signal, model_h: np.array, r: int, noise=None) -> np.array:
    f = FractalKalmanFilter(dim_x=r, dim_z=1)
    if noise is None:
        noise = np.zeros(len(signal))
    f_matrix = f.get_filter_matrix(r, model_h, len(signal))
    r_matrix = np.std(noise) ** 2
    h_matrix = f.H
    h_matrix[0][0] = 1.0
    f.set_matrices(h_matrix, r_matrix, f_matrix)

    estimated_signal = np.zeros(len(signal))
    for k in range(1, len(signal)):
        f.predict()
        if not np.isnan(signal[k]):
            f.update(signal[k])
        estimated_signal[k] = f.x[0].item()
    return estimated_signal


def apply_kalman_filter_cached(
    signal, model_h: np.array, r: int, cache_folder: Path, noise=None
) -> np.array:
    f = FractalKalmanFilter(dim_x=r, dim_z=1)
    if noise is None:
        noise = np.zeros(len(signal))
    file_path = cache_folder / f"{model_h}_{r}.npy"
    f_matrix = io.load_np_matrix(file_path)
    r_matrix = np.std(noise) ** 2
    h_matrix = f.H
    h_matrix[0][0] = 1.0
    f.set_matrices(h_matrix, r_matrix, f_matrix)

    estimated_signal = np.zeros(len(signal))
    for k in range(0, len(signal)):
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
    print("Prepare args...")
    args_list = []
    H_LIST = np.arange(0.5, 3.75, 0.25)
    R_LIST = [4]  # np.array([2**i for i in range(1, 6)])
    TRJ_LEN = 2**14
    n_times = 5
    s = TRJ_LEN
    SNR_LIST = [0.5]  # [0.1, 0.5, 1, 2]
    metrics_df = pd.DataFrame(
        columns=[
            "H_target",
            "H_signal",
            "H_estimated",
            "signal_len",
            "s",
            "r",
            "SNR",
            "SE",
            "H_kalman",
        ]
    )
    for snr in SNR_LIST:
        for h in H_LIST:
            args_list.append((h, s, R_LIST, TRJ_LEN, snr, n_times))
    print(f"Got {len(args_list) * len(R_LIST)} combinations for {n_times} times.")

    print("Run pool")
    with multiprocessing.Pool() as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_snr_h_iter, args_list),
                total=len(args_list),
                desc="Progress",
            )
        )

    print("Prepare results")
    for res in results:
        for row in res:
            metrics_df.loc[len(metrics_df)] = [
                row["H_target"],
                row["H_signal"],
                row["H_estimated"],
                row["signal_len"],
                row["s"],
                row["r"],
                row["SNR"],
                row["SE"],
                row["H_kalman"],
            ]
    file_name = "kalman-4.csv"
    metrics_df.to_csv(file_name, index=False)
    print(f"Metrics saved to {file_name}")
