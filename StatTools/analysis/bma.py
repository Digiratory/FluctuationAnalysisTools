import warnings
from collections.abc import Iterable
from typing import Tuple, Union

import numpy as np


def _bma_worker(
    array: np.ndarray,
    n: int,
    step: float = 1.0,
) -> np.ndarray:
    """
    Core Backward Moving Average (BMA) computation for a single scale n.
    Original articles:
    - Alessio, E., Carbone, A., Castelli, G. et al. Second-order moving average and scaling of stochastic time series. Eur. Phys. J. B 27, 197–200 (2002). https://doi.org/10.1140/epjb/e20020150
    - Carbone, A., Castelli, G., & Stanley, H. E. (2004). Analysis of clusters formed by the moving average of a long-range correlated time series. Physical Review E, 69(2), 026105. https://doi.org/10.1103/PhysRevE.69.026105

    For each time series y(t), this function:
      1. Computes the backward moving average ỹ(t) over past n points
         (no future values used).
      2. Computes the fluctuation function F(n) as the root mean square
         of [y(t) - ỹ(t)] over the available time indices.

    Args:
        array (np.ndarray):
            Time series array after optional integration.
            Shape: (n_signals, length).
        n (int):
            Window size (scale) for backward moving average.
        step (float):
            Fraction of scale n used for stepping between evaluation points
            (0 < step <= 1). For example:
              - step = 1.0   -> use every time index t
              - step = 0.5   -> use every (n * 0.5)-th index
            This controls the overlap between windows. Typical values are
            in [0.25, 1.0]. The actual integer step is:
                step_size = max(int(step * n), 1)

    Returns:
        np.ndarray:
            Fluctuation function values F(n) for each signal.
            Shape: (n_signals,).
    """
    n_signals, L = array.shape
    if n > L:
        raise ValueError(f"Window n={n} is larger than series length L={L}")

    # Indices t where we evaluate the fluctuation:
    #   t ∈ {n-1, n, ..., L-1} in 0-based indexing
    step_size = max(int(step * n), 1)
    t_indices = np.arange(n - 1, L, step_size)  # shape: (M,)

    if t_indices.size == 0:
        raise ValueError(
            f"No positions available for n={n} with given step={step}. "
            "Try smaller n or smaller step."
        )

    y = array  # (n_signals, L)

    # Cumulative sum for efficient window sums (along time axis).
    # cs[:, j] = sum_{i=0}^j y[:, i]
    cs = np.cumsum(y, axis=1)

    # For each t in t_indices, we need sum over [t-n+1 .. t]
    # Using cumulative sums:
    #   sum_{k=t-n+1}^t y[k] = cs[t] - cs[t-n] (for t-n >= 0)
    # NOTE: here t >= n-1, so t-n >= -1; for t-n == -1, sum = cs[t].
    window_sums = np.empty((n_signals, t_indices.size), dtype=float)

    t_indices = np.asarray(t_indices)

    # start и end для всех индексов
    start = t_indices - n + 1
    end = t_indices

    # Массив для результата
    window_sums = np.empty((cs.shape[0], len(t_indices)), dtype=cs.dtype)

    # Маски
    mask_start_le_zero = start <= 0
    mask_start_gt_zero = ~mask_start_le_zero

    # Обработка start <= 0
    if mask_start_le_zero.any():
        window_sums[:, mask_start_le_zero] = cs[:, end[mask_start_le_zero]]

    # Обработка start > 0
    if mask_start_gt_zero.any():
        window_sums[:, mask_start_gt_zero] = (
            cs[:, end[mask_start_gt_zero]] - cs[:, start[mask_start_gt_zero] - 1]
        )

    y_mean = window_sums / float(n)  # backward moving average, shape (n_signals, M)

    # Values of y(t) at t_indices
    y_t = y[:, t_indices]  # shape (n_signals, M)

    # Residuals and fluctuation function:
    #   F(n) = sqrt( mean_t [ y(t) - ỹ(t) ]^2 )
    residuals = y_t - y_mean
    F_n = np.sqrt(np.mean(residuals**2, axis=1))  # shape (n_signals,)

    return F_n


def bma(
    arr: np.ndarray,
    s: Union[int, Iterable[int]],
    n_integral: int = 1,
    step: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Execute Backward Detrended Moving Average (BMA/DMA) analysis for time series.

    Original articles:
    - Alessio, E., Carbone, A., Castelli, G. et al. Second-order moving average and scaling of stochastic time series. Eur. Phys. J. B 27, 197–200 (2002). https://doi.org/10.1140/epjb/e20020150
    - Carbone, A., Castelli, G., & Stanley, H. E. (2004). Analysis of clusters formed by the moving average of a long-range correlated time series. Physical Review E, 69(2), 026105. https://doi.org/10.1103/PhysRevE.69.026105

    This method implements the algorithm:

        1. (Optional) Integration
           If n_integral >= 1, the input series x(t) is integrated n_integral times:
               y(t) = cumsum(x(t))          # one integration
               y(t) = cumsum(cumsum(...))   # repeated n_integral times
           For n_integral = 1:
             - original x(t) is interpreted as fractional Gaussian noise (fGn),
             - integrated y(t) corresponds to fractional Brownian motion (fBm).

        2. Backward moving average (BMA)
           For each scale n (window size), compute the backward moving average:
               ỹ(t) = (1/n) * sum_{k=0}^{n-1} y(t - k),
           where t runs over valid indices (t >= n-1 in 0-based indexing),
           and only past values are used (no future samples).
    3. Fluctuation function
           The fluctuation function is defined as:
               F(n) = sqrt( (1/M) * sum_t [ y(t) - ỹ(t) ]^2 ),
           where the sum is taken over selected indices t with step size
           proportional to n (controlled by the 'step' parameter), and M is
           the number of such indices.

        4. Hurst exponent estimation
           Under a power-law scaling:
               F(n) ~ n^H,
           the Hurst exponent H can be estimated as the slope of a linear
           regression of log F(n) vs log n over a chosen range of scales.

    Basic usage:
        import numpy as np

        # Single time series
        data = np.random.normal(0, 1, 10000)
        scales = [16, 32, 64, 128, 256]
        F, used_scales = bma(data, s=scales, n_integral=1, step=0.5)

        # Multiple time series
        data_multi = np.random.normal(0, 1, (5, 10000))  # 5 signals
        F_multi, used_scales = bma(data_multi, s=scales, n_integral=1, step=0.5)

        # Estimation of H:
        #   Fit a line to (log(used_scales), log(F)) and take the slope.

    Args:
        arr (np.ndarray):
            Input time series data.
            - 1D array: single time series of shape (length,)
            - 2D array: multiple time series of shape (n_signals, length)
        s (Union[int, Iterable[int]]):
            Scale values (window sizes n) for analysis.
            - Single int: analyze at one scale
            - List/array of ints: analyze at multiple scales
            Scales larger than length / 4 are filtered out.
        n_integral (int):
            Number of cumulative sum operations (integrations) to apply.
            - 0: no integration, treat arr as y(t) directly
            - 1: single integration (typical for fGn -> fBm)
            - 2+: repeated integration
        step (float):
            Fraction of scale n for stepping between evaluation points (0 < step <= 1).
            Actual integer step is:
                step_size = max(int(step * n), 1).
            This controls overlap between windows. Typical values: 0.25, 0.5, 1.0.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            (F_values, scales), where:
              - F_values:
                  * Single signal: shape (n_scales,)
                  * Multiple signals: shape (n_signals, n_scales)
                Contains fluctuation function values F(n) for each scale.
              - scales:
                  Array of actually used scale values (after filtering by length / 4).

    Raises:
        ValueError:
            - If input array dimension > 2
            - If all scale values are larger than length / 4
            - If a given scale produces no evaluation positions with the chosen step
    """
    # Ensure correct dimensionality
    if len(arr.shape) > 2:
        raise ValueError(
            f"Unsupported dimension of input signals array: "
            f"expected 1 or 2, got shape {arr.shape}"
        )

    # Bring data to shape (n_signals, length)
    if arr.ndim == 1:
        y = arr[np.newaxis, :]
    else:
        y = arr

    n_signals, L = y.shape

    # Validate and filter scales
    if isinstance(s, Iterable) and not isinstance(s, (str, bytes)):
        s = list(s)
        init_s_len = len(s)
        s = list(filter(lambda x: x <= L / 4, s))
        if len(s) < 1:
            raise ValueError(
                "All input scale values are larger than series length / 4!"
            )
        if len(s) != init_s_len:
            warnings.warn(f"\tBMA warning: only following scales are in use: {s}")
    elif isinstance(s, (float, int)):
        if s > L / 4:
            raise ValueError("Cannot use scale > length / 4")
        s = [int(s)]
    else:
        raise ValueError("Unsupported type for 's'. Provide int or iterable of ints.")

    # Optional integration (n_integral times)
    for _ in range(max(int(n_integral), 0)):
        y = np.cumsum(y, axis=1)

    # Compute F(n) for each scale
    F = np.zeros((n_signals, len(s)), dtype=float)
    for idx, n in enumerate(s):
        n_int = int(n)
        if n_int < 1:
            raise ValueError(f"Scale values must be >= 1, got {n}")
        F[:, idx] = _bma_worker(y, n_int, step=step)

    # Return with consistent shape for 1D vs 2D input
    if arr.ndim == 1:
        return F[0], np.array(s, dtype=int)
    return F, np.array(s, dtype=int)
