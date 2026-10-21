"""
Gap distortion + stochastic Kalman reconstruction experiment.

For every (model_h, signal_idx, rq, fg) combination, a single random gap
mask is drawn once and shared across every filter order in FILTER_ORDERS, so
the R_Q comparison between orders is not confounded by different gap
realisations. Both the gapped and reconstructed signals are cached to disk
so repeated runs (e.g. after growing the parameter grid) skip finished work.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import numpy.typing as npt
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from research.kalman_filter.reconstruction_of_long_term_complex_system_dynamics.gen_filter_save import (
    simulate_signal,
)
from StatTools.experimental.augmentation.perturbations import add_exponential_gaps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

NDArrayF64 = npt.NDArray[np.float64]

_ORIG_CACHE: dict = {}
_NOISE_CACHE: dict = {}


def _get_original_and_noise(
    signals_dir: Path, model_h: float, signal_idx: int
) -> tuple[NDArrayF64, NDArrayF64]:
    key = (model_h, signal_idx)
    if key not in _ORIG_CACHE:
        original = np.load(signals_dir / f"original_hm{model_h}_{signal_idx}.npy")
        _ORIG_CACHE[key] = original
        _NOISE_CACHE[key] = np.random.default_rng(signal_idx).normal(
            0, np.nanstd(original) * 1e-6, size=original.shape
        )
    return _ORIG_CACHE[key], _NOISE_CACHE[key]


def get_rq(signal: NDArrayF64) -> float:
    """Mean spacing between zero-crossings of the median-centered signal."""
    s = signal - np.nanmedian(signal)
    crossings = np.flatnonzero((s[:-1] < 0) & (s[1:] >= 0)) + 1
    return float(np.mean(np.diff(crossings)))


def _process_gap_combination(
    model_h: float,
    signal_idx: int,
    rq: float,
    fg: float,
    filter_orders: list[int],
    method: str,
    signals_dir: Path,
    gaps_dir: Path,
) -> list[dict]:
    """Distort one signal once, reconstruct it at every order, return metric rows."""

    original, tiny_noise = _get_original_and_noise(signals_dir, model_h, signal_idx)

    gapped_path = gaps_dir / f"gapped_hm{model_h}_{signal_idx}_rq{rq}_fg{fg}.npy"
    if gapped_path.exists():
        gapped = np.load(gapped_path)
    else:
        gapped, _ = add_exponential_gaps(original, rq=rq, fg=fg)
        np.save(gapped_path, gapped)
    gap_mask = np.isnan(gapped)

    rq_original = get_rq(original)
    rq_gapped = get_rq(gapped)

    rows = []
    for order in filter_orders:
        recovered_path = (
            gaps_dir
            / f"reconstructed_hm{model_h}_{signal_idx}_rq{rq}_fg{fg}_{method}_o{order}.npy"
        )
        if recovered_path.exists():
            recovered = np.load(recovered_path)
        else:
            recovered = simulate_signal(
                gapped, tiny_noise, model_h, method, order, seed=signal_idx
            )
            np.save(recovered_path, recovered)

        rmse_gap = float(
            np.sqrt(np.mean((recovered[gap_mask] - original[gap_mask]) ** 2))
        )
        rows.append(
            {
                "model_h": model_h,
                "signal_idx": signal_idx,
                "rq_gap": rq,
                "fg_gap": fg,
                "order": order,
                "rq_original": rq_original,
                "rq_gapped": rq_gapped,
                "rq_reconstructed": get_rq(recovered),
                "rmse_gap": rmse_gap,
                "method": method,
            }
        )
    return rows


def run_gap_reconstruction_experiment(
    h_list: list[float],
    signals_count: int,
    filter_orders: list[int],
    rq_gap_list: list[float],
    fg_gap_list: list[float],
    method: str,
    signals_dir: Path,
    gaps_dir: Path,
    csv_dir: Path,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    For every (model_h, signal_idx, rq, fg):
      1. Draw one gap mask (shared by every order below) and cache it.
      2. Reconstruct with simulate_signal at each order in filter_orders.
      3. Record R_Q of the original vs. R_Q of each reconstruction.

    Returns the accumulated metrics as a DataFrame and also saves them to
    gaps_dir / f"metrics_rq_{method}.csv".
    """
    gaps_dir.mkdir(parents=True, exist_ok=True)
    results = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
        delayed(_process_gap_combination)(
            model_h, signal_idx, rq, fg, filter_orders, method, signals_dir, gaps_dir
        )
        for model_h in h_list
        for signal_idx in range(signals_count)
        for rq in rq_gap_list
        for fg in fg_gap_list
    )

    rows = []
    for chunk in tqdm(
        results,
        total=len(h_list) * signals_count * len(rq_gap_list) * len(fg_gap_list),
        desc="Reconstructing",
        unit="combo",
    ):
        rows.extend(chunk)

    metrics_df = pd.DataFrame(rows)
    metrics_path = csv_dir / f"metrics_rq_{method}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    log.info("Metrics saved to %s", metrics_path)

    return metrics_df


# Entry point

if __name__ == "__main__":
    SIGNALS_DIR = Path(
        "research/kalman_filter/reconstruction_of_long_term_complex_system_dynamics/data/model_signals"
    )
    GAPS_DIR = Path("research/kalman_filter/gaps_research/data/model_signals_gaps")
    CSV_DIR = Path("research/kalman_filter/gaps_research/data")

    H_LIST: list[float] = [0.7, 0.8, 0.9, 1.0]
    SIGNALS_COUNT: int = 8
    FILTER_ORDERS: list[int] = [2, 4, 12]
    RQ_GAP_LIST: list[float] = [5, 10, 30, 50, 75, 100]
    FG_GAP_LIST: list[float] = [0.05, 0.1, 0.3]
    METHOD = "YW"

    run_gap_reconstruction_experiment(
        H_LIST,
        SIGNALS_COUNT,
        FILTER_ORDERS,
        RQ_GAP_LIST,
        FG_GAP_LIST,
        METHOD,
        SIGNALS_DIR,
        GAPS_DIR,
        CSV_DIR,
        n_jobs=16,
    )
