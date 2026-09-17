"""
Signal generation, noise perturbation, and Kalman filtering pipeline.
"""

from __future__ import annotations

import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from scipy import stats
from statsmodels.regression.linear_model import yule_walker
from tqdm import tqdm

from StatTools.analysis.dfa import dfa
from StatTools.experimental.augmentation.perturbations import add_noise
from StatTools.experimental.filters.symbolic_kalman import (
    get_sympy_filter_matrix,
    refine_filter_matrix,
)
from StatTools.filters.kalman_filter import KalmanFilter
from StatTools.generators import generate_fbn
from StatTools.generators.kasdin_generator import create_kasdin_generator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

NDArrayF64 = npt.NDArray[np.float64]


_F_CACHE: dict[tuple, NDArrayF64] = {}
_Q_CACHE: dict[tuple, NDArrayF64] = {}


def get_F_kinematic(order: int) -> NDArrayF64:
    """Constant-velocity (uniform rectilinear motion) state-transition matrix,
    generalised to `order` states via a discrete Taylor expansion (dt=1).

    order=2 is the textbook constant-velocity model [[1, 1], [0, 1]]
    (position += velocity, velocity unchanged); higher orders extend it to
    constant acceleration, constant jerk, etc.:
        F[i, j] = 1/(j-i)!  for j >= i, else 0
    (dt=1, so the usual dt^(j-i) factor drops out). Unlike Kasdin/YW/
    BackDiff, this fits no AR/fBm structure — it assumes a fixed-order
    polynomial (deterministic) trend, with correction coming only from Q/R
    via the Kalman gain. It is the reference filter against which the
    model-based F constructions above are judged.
    """
    F = np.zeros((order, order))
    for i in range(order):
        for j in range(i, order):
            F[i, j] = 1.0 / math.factorial(j - i)
    return F


def get_F_yw(signal: np.ndarray, order: int) -> np.ndarray:
    """
    Yule-Walker AR(order) estimate → companion matrix F.

    Uses moment-matching (Toeplitz system) via statsmodels.
    Unlike OLS, YW guarantees a valid autocorrelation-consistent solution.
    """
    ar_coeffs, _ = yule_walker(signal, order=order, method="mle")

    # Companion matrix:
    # [ a1  a2  ...  a_{p-1}  ap ]
    # [  1   0  ...      0     0 ]
    # [  0   1  ...      0     0 ]
    # [         ...              ]
    F = np.zeros((order, order))
    F[0, :] = ar_coeffs
    F[1:, :-1] = np.eye(order - 1)
    return F


def _compute_h(sig):
    sig = np.array(sig)
    s_vals, f2_vals = dfa(sig, degree=2, processes=1)
    log_s = np.log(s_vals)
    log_f = np.log(np.sqrt(f2_vals))
    return round(stats.linregress(log_s, log_f).slope, 3)


def _cached_F_matrix(model_h: float, order: int, length: int) -> NDArrayF64:
    key = (model_h, order, length)
    if key not in _F_CACHE:
        generator = create_kasdin_generator(
            model_h,
            length=length,
            filter_type="lfilter_truncated",
            filter_coefficients_length=order + 1,
        )
        ar_filter = generator.get_filter_coefficients()
        mat = refine_filter_matrix(get_sympy_filter_matrix(order), order, ar_filter)
        _F_CACHE[key] = np.array(mat, dtype=np.float64)
    return _F_CACHE[key]


def _cached_Q_matrix(model_h: float, order: int, length: int) -> NDArrayF64:
    key = (model_h, order, length)
    if key not in _Q_CACHE:
        signal = generate_fbn(
            hurst=model_h,
            length=length,
            method="kasdin",
            filter_type="lfilter_truncated",
        )[0]
        ridge = 1e-6 * np.var(signal)
        Q = np.eye(order) * ridge
        Q[0, 0] = np.var(signal)
        _Q_CACHE[key] = Q
    return _Q_CACHE[key]


def _get_H(order: int) -> NDArrayF64:
    H = np.zeros(order)
    H[0] = 1.0
    return H


def _build_kalman_system(
    noisy_signal: NDArrayF64,
    noise: NDArrayF64,
    model_h: float,
    method: str,
    order: int,
) -> tuple[NDArrayF64, NDArrayF64, NDArrayF64, NDArrayF64]:
    length = len(noisy_signal)

    if method == "Kasdin":
        F = _cached_F_matrix(model_h, order, length)
    elif method == "YW":
        # for purity, not the original signal is used, but a synthetic signal with a given H.
        signal = generate_fbn(
            hurst=model_h,
            length=len(noisy_signal),
            method="kasdin",
            filter_type="lfilter_truncated",
        )[0]
        F = get_F_yw(signal, order)
    elif method == "Baseline":
        F = get_F_kinematic(order)
    else:
        raise ValueError(f"Unknown Kalman filter method: {method!r}")

    H = _get_H(order)
    R = np.array([[np.nanvar(noise)]])
    Q = _cached_Q_matrix(model_h, order, length)
    return F, H, R, Q


def filter_signal(
    noisy_signal: NDArrayF64,
    noise: NDArrayF64,
    model_h: float,
    method: str,
    order: int,
) -> NDArrayF64:
    F, H, R, Q = _build_kalman_system(noisy_signal, noise, model_h, method, order)
    kf = KalmanFilter(order, 1, F=F, H=H, R=R, Q=Q)

    recovered = np.empty(len(noisy_signal), dtype=np.float64)
    for i, sample in enumerate(noisy_signal):
        kf.predict()
        kf.adjust(np.array([[sample]]))
        recovered[i] = kf.get_current_measurement().item()

    return recovered


# Single-job helpers (called in parallel)


def _generate_one(
    model_h: float,
    signal_idx: int,
    signal_len: int,
    data_dir: Path,
) -> None:
    """Generate and save one signal realisation."""
    signal_path = data_dir / f"original_hm{model_h}_{signal_idx}.npy"
    if signal_path.exists():
        return
    h_dif = 1
    while h_dif > 0.05:
        signal = generate_fbn(
            hurst=model_h,
            length=signal_len,
            method="kasdin",
            filter_type="lfilter_truncated",
            seed=signal_idx,
        )[0]
        h_dif = np.abs(model_h - _compute_h(signal))
        print(f"h_dif: {h_dif:.3f}")
    np.save(signal_path, signal)


def _process_one_combination(
    signal_path: Path,
    model_h: float,
    signal_idx: int,
    noise_ratio: float,
    method: str,
    order: int,
    data_dir: Path,
) -> None:
    """Load signal from disk, add noise, filter, and save."""
    noisy_path = data_dir / f"noisy_hm{model_h}_{signal_idx}_nr{noise_ratio}.npy"
    signal = np.load(signal_path)

    time.sleep(random.random() * 3)  # Sleep randomly up to 3 seconds
    # It is redundunt, but conditioned execution may cause overlapping
    noisy_signal, noise = add_noise(signal, ratio=noise_ratio, noise_seed=signal_idx)
    if Path(noisy_path).exists():
        noisy_signal = np.load(noisy_path)
        noise = noisy_signal - signal
    else:
        np.save(noisy_path, noisy_signal)

    for filter_h in H_LIST:
        recovered_path = (
            data_dir
            / f"recovered_hm{model_h}_{signal_idx}_nr{noise_ratio}_{method}_o{order}_hf{filter_h}.npy"
        )
        recovered = filter_signal(noisy_signal, noise, filter_h, method, order)
        np.save(
            recovered_path,
            recovered,
        )


# Public pipeline functions


def generate_save_signals(
    h_list: list[float],
    signal_len: int,
    signals_count: int,
    data_dir: Path,
    n_jobs: int = -1,
) -> None:
    """Generate all realisations for all h values in parallel."""
    jobs = [
        delayed(_generate_one)(h, idx, signal_len, data_dir)
        for h in h_list
        for idx in range(signals_count)
    ]
    log.info(
        "Generating %d signals (%d h-values x %d realisations)...",
        len(jobs),
        len(h_list),
        signals_count,
    )
    Parallel(n_jobs=n_jobs)(tqdm(jobs, desc="Generating", unit="signal"))
    log.info("Generation complete. Signals saved to %s", data_dir)


def add_noise_filter_save_signals(
    data_dir: Path,
    signals_count: int,
    h_list: list[float],
    noise_list: list[float],
    methods: list[str],
    orders: list[int],
    n_jobs: int = -1,
) -> None:
    """
    For every combination of (model_h, signal_idx, noise_ratio, method, order):
      1. Load the pre-generated signal from disk (inside each worker).
      2. Add noise.
      3. Filter with the specified Kalman method and order.
      4. Save the recovered signal.
    """
    jobs = []
    for model_h in h_list:
        for signal_idx in range(signals_count):
            signal_path = data_dir / f"original_hm{model_h}_{signal_idx}.npy"
            for noise_ratio in noise_list:
                for method in methods:
                    for order in orders:
                        jobs.append(
                            delayed(_process_one_combination)(
                                signal_path,
                                model_h,
                                signal_idx,
                                noise_ratio,
                                method,
                                order,
                                data_dir,
                            )
                        )

    log.info("Dispatching %d filter jobs (n_jobs=%s)...", len(jobs), n_jobs)
    Parallel(n_jobs=n_jobs)(tqdm(jobs, desc="Filtering", unit="job"))
    log.info("All jobs complete.")


# Entry point

if __name__ == "__main__":
    # Paths
    SIGNALS_DIR = Path(
        "research/kalman_filter/reconstruction_of_long_term_complex_system_dynamics/data/model_signals"
    )
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

    # Generation parameters
    H_LIST: list[float] = [0.7, 0.8, 0.9, 1.0]
    SIGNALS_COUNT: int = 1  # number of realisations per h value
    SIGNAL_LENGTH: int = 2**10  # samples

    # Perturbation parameters
    NOISE_RATIOS: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9, 2.0, 3.0]

    # Filtering parameters
    METHODS: list[str] = ["Kasdin"]
    ORDERS: list[int] = [2, 4, 8]

    # Generation
    generate_save_signals(
        H_LIST,
        SIGNAL_LENGTH,
        SIGNALS_COUNT,
        SIGNALS_DIR,
        n_jobs=16,
    )

    # Perturbation + Filtering
    add_noise_filter_save_signals(
        SIGNALS_DIR,
        SIGNALS_COUNT,
        H_LIST,
        NOISE_RATIOS,
        METHODS,
        ORDERS,
        n_jobs=16,
    )
