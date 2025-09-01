from typing import Iterable, Tuple, Union

import numpy as np


def _fa_worker(array: np.ndarray, step: float, S: float) -> np.ndarray:
    """
    Core Fluctuation Analysis (FA) computation for a single scale.

    Computes the fluctuation function F²(s) for integrated time series at a given scale s.
    The method calculates the variance of integrated signal differences over windows
    of size s, providing a measure of signal fluctuations at that scale.

    Args:
        array (np.ndarray): Preprocessed integrated time series array.
            Shape: (n_signals, length) for multiple signals.
        step (float): Fraction of scale s used for window stepping (0 < step <= 1).
            Typically 0.5 for 50% overlap between windows.
        S (float): Scale (window size) for fluctuation computation.

    Returns:
        np.ndarray: Fluctuation function values F²(s) for each signal.
            Shape: (n_signals,) - one value per input signal.
    """
    V = np.arange(0, array.shape[1] - S, max(int(step * S), 1))
    Fv = np.zeros((array.shape[0], len(V)), dtype=float)
    for v_i, v in enumerate(V):
        Fv[:, v_i] = array[:, v] - array[:, v + S]
    F2 = np.mean(Fv**2, axis=1)
    return F2


def fa(
    arr: np.ndarray, step: float, s: Union[int, Iterable], n_integral=1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Execute Fluctuation Analysis (FA) for time series data.

    FA computes the fluctuation function F²(s) which measures the variance of
    integrated signal differences over windows of size s. This method is useful
    for detecting scaling behavior and long-range correlations in time series.

    The fluctuation function typically follows a power-law relationship:
    F²(s) ~ s^{2H} where H is the Hurst exponent.

    Basic usage:
        ```python
        import numpy as np
        from StatTools.analysis.fa import fa

        # Single time series
        data = np.random.normal(0, 1, 10000)
        scales = [16, 32, 64, 128, 256]
        F_squared, used_scales = fa(data, step=0.5, s=scales)

        # Multiple time series
        data_multi = np.random.normal(0, 1, (5, 10000))  # 5 signals
        F_squared_multi, used_scales = fa(data_multi, step=0.5, s=scales)
        ```

    Args:
        arr (np.ndarray): Input time series data. Can be:
            - 1D array: Single time series (length,)
            - 2D array: Multiple time series (n_signals, length)
        step (float): Fraction of scale s for window stepping (0 < step <= 1).
            Controls overlap between analysis windows. Typical value: 0.5.
        s (Union[int, Iterable]): Scale values for analysis. Can be:
            - Single int: Analyze at one scale
            - List/array: Analyze at multiple scales
            Scales must be <= length/4 to ensure sufficient data.
        n_integral (int): Number of cumulative sum operations to apply.
            Default 1 performs single integration. Use 2 for double integration.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (F_squared, scales) where:
            - F_squared: Fluctuation function F²(s) values
                - Single signal: shape (n_scales,)
                - Multiple signals: shape (n_signals, n_scales)
            - scales: Array of scale values actually used in analysis

    Raises:
        ValueError: If input array dimensions > 2 or scale values are invalid

    Note:
        Scales larger than length/4 are automatically filtered out with a warning.
        The method applies cumulative summation (integration) before fluctuation analysis.
    """
    if len(arr.shape) > 2:
        raise ValueError(
            f"Unsupported dimention of input signals array: expected 1 or 2, got {arr.shape}"
        )

    if len(arr.shape) == 1:
        cumsum_arr = arr[np.newaxis, :]
    else:
        cumsum_arr = arr

    if isinstance(s, Iterable):
        init_s_len = len(s)

        s = list(filter(lambda x: x <= cumsum_arr.shape[1] / 4, s))
        if len(s) < 1:
            raise ValueError("All input S values are larger than vector shape / 4 !")

        if len(s) != init_s_len:
            print(f"\tFA warning: only following S values are in use: {s}")

    elif isinstance(s, (float, int)):
        if s > cumsum_arr.shape[1] / 4:
            raise ValueError("Cannot use S > L / 4")
        s = (s,)

    s_current = [s] if not isinstance(s, Iterable) else s

    for _ in range(n_integral):
        cumsum_arr = np.cumsum(cumsum_arr, axis=1)

    F = np.zeros((cumsum_arr.shape[0], len(s_current)), dtype=float)

    for s_idx, s in enumerate(s_current):
        F[:, s_idx] = _fa_worker(cumsum_arr, step, s)

    return (F[0] if len(arr.shape) == 1 else F, s_current)
