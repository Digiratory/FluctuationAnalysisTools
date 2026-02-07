from __future__ import annotations

import warnings
from contextlib import closing
from functools import partial
from math import exp, floor
from multiprocessing import Pool, cpu_count
from typing import Tuple, Union

import numpy as np

_CUPY_WARNED = False


# ====================== DFA core worker ======================


def dfa_worker(
    indices: Union[int, list, np.ndarray],
    arr: Union[np.ndarray, None] = None,
    degree: int = 2,
    s_values: Union[list, np.ndarray, None] = None,
    n_integral: int = 1,
    backend: str = "cpu",
) -> list:
    """
    Core of the DFA algorithm. Processes a subset of series (indices) and
    returns fluctuation functions F^2(s).

    Args:
        indices: Indices of time series in the dataset to process.
        arr: Dataset array (must be 2D, shape: (n_series, length)).
        degree: Polynomial degree for detrending.
        s_values: Pre-calculated box sizes (scales).
        n_integral: Number of cumulative sum operations to apply (default: 1).
        backend: Computational backend ("cpu" or "gpu").

    Returns:
        list of (s, F2_s) for each requested index, where F2_s is F^2(s).
    """

    data = np.asarray(arr, dtype=float)
    if data.ndim != 2:
        raise ValueError(
            f"dfa_worker expects 2D array, got {data.ndim}D array. "
            f"Normalize data to 2D before calling (use reshape(1, -1) for 1D)."
        )

    indices = np.atleast_1d(indices).astype(int).ravel()
    n_series = data.shape[0]
    for idx in indices:
        if not (0 <= idx < n_series):
            raise IndexError(
                f"Index {idx} out of bounds for array with {n_series} series"
            )

    series_len_global = data.shape[1]

    if s_values is None:
        s_max = int(series_len_global / 4)
        s_values = [int(exp(step)) for step in np.arange(1.6, np.log(s_max), 0.5)]
    else:
        if isinstance(s_values, np.ndarray):
            s_values = s_values.tolist()
        s_values = list(s_values)

    if backend == "cpu":
        xp = np
    elif backend == "gpu":
        try:
            import cupy as cp  # type: ignore
        except ImportError:
            global _CUPY_WARNED
            if not _CUPY_WARNED:
                warnings.warn(
                    "CuPy not available, GPU acceleration disabled. "
                    "Switching to CPU backend. "
                    "Install with: pip install cupy-cuda11x (or cupy-cuda12x)",
                    UserWarning,
                    stacklevel=2,
                )
                _CUPY_WARNED = True
            backend = "cpu"
            xp = np
        else:
            xp = cp
    else:
        raise ValueError(f'backend must be "cpu" or "gpu", got: {backend!r}')

    results = []

    for idx in indices:
        series = xp.asarray(data[idx], dtype=xp.float64)

        if series.ndim > 1:
            series = series.flatten()

        data_centered = series - xp.mean(series)
        y_cumsum = data_centered
        for _ in range(n_integral):
            y_cumsum = xp.cumsum(y_cumsum)

        series_len = int(series.shape[0])

        s_list = []
        f2_list = []

        for s_val in s_values:
            if s_val >= series_len / 4:
                continue

            s = xp.arange(1, s_val + 1, dtype=int)
            cycles_amount = floor(series_len / s_val)

            if cycles_amount < 1:
                continue

            f2_sum = 0.0
            s_temp = s.copy()

            for i in range(1, cycles_amount):
                indices_s = (s_temp - (i + 0.5) * s_val).astype(int)
                y_cumsum_s = y_cumsum[s_temp]

                coef = xp.polyfit(indices_s, y_cumsum_s, deg=degree)
                trend = xp.polyval(coef, indices_s)
                residuals = y_cumsum_s - trend

                f2 = xp.sum(residuals**2) / s_val
                f2_sum += f2
                s_temp += s_val

            f2_s = f2_sum / (cycles_amount - 1)
            s_list.append(s_val)
            f2_list.append(f2_s)

        s_array = np.array(s_list)
        if backend == "gpu":
            f2_array = xp.asarray(f2_list).get()
        else:
            f2_array = np.array(f2_list)

        results.append((s_array, f2_array))

    return results


# ====================== High-level DFA function ======================


def dfa(
    dataset,
    degree: int = 2,
    processes: int = 1,
    n_integral: int = 1,
    backend: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementation of the Detrended Fluctuation Analysis (DFA) method.

    The algorithm removes local polynomial trends in integrated time series and
    returns the fluctuation function F^2(s) for each series.

    Args:
        dataset (ndarray): 1D or 2D array of time series data.
        degree (int): Polynomial degree for detrending (default: 2).
        processes (int): Number of parallel workers (default: 1).
            Note: If backend="gpu", multiprocessing is disabled and GPU is used instead.
        n_integral (int): Number of cumulative sum operations to apply (default: 1).
        backend (str): Computational backend. Options: "cpu" (default), "gpu".
            GPU mode uses single process (multiprocessing disabled).

    Returns:
        tuple: (s, F2_s)
            - For 1D input: two 1D arrays s, F2_s.
            - For 2D input:
                s is a 1D array (same scales for all series),
                F2_s is a 2D array where each row is F^2(s) for one time series.
    """
    data = np.asarray(dataset, dtype=float)
    if data.size == 0:
        raise ValueError("Input dataset is empty.")

    if data.ndim == 1:
        data = data.reshape(1, -1)
        single_series = True
    elif data.ndim == 2:
        single_series = False
    else:
        raise ValueError("Only 1D or 2D arrays are allowed!")

    series_len = data.shape[1]
    s_max = int(series_len / 4)
    s_values = [int(exp(step)) for step in np.arange(1.6, np.log(s_max), 0.5)]

    n_series = data.shape[0]

    # Validate backend
    if backend not in ("cpu", "gpu"):
        raise ValueError(f'backend must be "cpu" or "gpu", got: {backend!r}')

    if backend == "gpu" and processes > 1:
        warnings.warn(
            f"GPU acceleration enabled: multiprocessing (processes={processes}) "
            f"is disabled. Using single GPU process instead.",
            UserWarning,
        )

    if backend == "gpu" or processes <= 1:
        indices = np.arange(n_series)
        results = dfa_worker(
            indices=indices,
            arr=data,
            degree=degree,
            s_values=s_values,
            n_integral=n_integral,
            backend=backend,
        )
    else:
        processes = min(processes, cpu_count(), n_series)
        chunks = np.array_split(np.arange(n_series), processes)

        worker_func = partial(
            dfa_worker,
            arr=data,
            degree=degree,
            s_values=s_values,
            n_integral=n_integral,
            backend="cpu",
        )

        results_list_of_lists = []
        with closing(Pool(processes=processes)) as pool:
            for sub in pool.map(worker_func, chunks):
                results_list_of_lists.append(sub)

        results = [item for sub in results_list_of_lists for item in sub]

    s_list = [r[0] for r in results]
    f2_list = [r[1] for r in results]

    if single_series:
        s_out = s_list[0]
        f2_out = f2_list[0]
    else:
        s_out = s_list[0]
        f2_out = np.vstack(f2_list)

    return s_out, f2_out
