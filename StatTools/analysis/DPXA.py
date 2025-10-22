from typing import Tuple, Union

import numpy as np
from analysis import dpcca


def dpxa(
    arr: np.ndarray, pd: int, step: float, s: Union[int, list, np.ndarray], maxlag: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    DPCCA with time lags: DPXA.
    Returns:
        P_lag: [N_signals-coordinates of p, N_signals- value of signals, len(s)] — maximum cross correlation with definite time lag,
        T_lag: [N_signals, N_signals, len(s)] — the best time lag tau with maximum cross correlation,
        S: time scales.
    Input: N- data coordinates of signals and coordinates of signals with time lag
        S-array of time scales
        L-time points
        num_scales- value of time scales
        lags- time lags: lags=(2*lags-1)
    Raises:
        ValueError: All input S values are larger than vector shape / 4.
        ValueError: Cannot use S > L / 4.
    P_tau-array of cross correlation for N coordinates and N+tau/N-tau coordinates for each time scale
    P_tau- in array with signal's data with realization in each time lag finds fluctuation function
    between N and N+-tau lag: cross correlation
    best_lag_idx - time lag where cross correlation is maximum in each time scale
    """

    L = arr.shape[1]
    s = [si for si in s if si <= L // 4]
    if not s:
        raise ValueError("s is very big")

    N = arr.shape[0]
    num_scales = len(s)
    lags = np.arange(-maxlag - 1, maxlag)

    P_lag = np.full((N, N, num_scales), np.nan)
    T_lag = np.full((N, N, num_scales), np.nan)

    for idx_s, S_val in enumerate(s):
        P_tau = np.full((N, N, len(lags)), np.nan)

        for lag_idx, tau in enumerate(lags):
            X_shifted = array_with_delay(arr, tau)
            try:
                P_ij, _, _, _ = dpcca([S_val], X_shifted, step, pd)
                P_tau[:, :, lag_idx] = P_ij[0]
            except Exception as e:
                print(f"Error at S={S_val}, tau={tau}: {e}")
                continue

        best_lag_idx = np.argmax(np.abs(P_tau), axis=2)
        P_lag[:, :, idx_s] = np.take_along_axis(
            P_tau, best_lag_idx[..., None], axis=2
        ).squeeze(axis=-1)
        """
        P_lag-array with maximum cross correlations
        T_lag - new 2d array with time lags with max cross correlation

        """
        T_lag[:, :, idx_s] = lags[best_lag_idx]

    return P_lag, T_lag, None, s


def array_with_delay(arr: np.ndarray, tau: int) -> np.ndarray:
    """
    Returns:
        New array with time lags, in new array shifted data is Nan

    if tau > 0: signal is delayed -> data is cut off at the front.
    if tau < 0: signal is ahead -> data is cut off at the back.
    """
    if tau == 0:
        return arr.copy()
    shifted = np.full_like(arr, np.nan)
    for i in range(arr.shape[0]):
        if tau > 0:
            shifted[i, tau:] = arr[i, :-tau]
        elif tau < 0:
            shifted[i, :tau] = arr[i, -tau:]

    return shifted
