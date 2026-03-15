import gc
import warnings
from collections.abc import Iterable
from contextlib import closing
from ctypes import c_double
from functools import partial
from multiprocessing import Pool
from typing import Union

import numpy as np


def _covariation_single_signal(signal: np.ndarray):
    """
    Implementation equation (4) from [1]

    [1] Yuan, N., Fu, Z., Zhang, H. et al. Detrended Partial-Cross-Correlation Analysis: A New Method for Analyzing Correlations in Complex System. Sci Rep 5, 8143 (2015). https://doi.org/10.1038/srep08143
    """
    F = np.zeros((signal.shape[0], signal.shape[0]), dtype=float)
    for n in range(signal.shape[0]):
        for m in range(n + 1):
            F[n][m] = np.mean(signal[n] * signal[m])
            F[m][n] = F[n][m]
    return F


def _covariation(signal_1: np.ndarray, signal_2: np.ndarray = None):
    """
    Implementation equation (4) from [1]

    [1] Yuan, N., Fu, Z., Zhang, H. et al. Detrended Partial-Cross-Correlation Analysis: A New Method for Analyzing Correlations in Complex System. Sci Rep 5, 8143 (2015). https://doi.org/10.1038/srep08143
    """
    if signal_2 is None:
        return _covariation_single_signal(signal_1)
    F = np.zeros((signal_1.shape[0], signal_1.shape[0]), dtype=float)
    for n in range(signal_1.shape[0]):
        for m in range(signal_2.shape[0]):
            F[n][m] = np.mean(signal_1[n] * signal_2[m])
    return F


def _correlation(F: np.ndarray):
    """
    Implementation equation (6) from [1]
    [1] Yuan, N., Fu, Z., Zhang, H. et al. Detrended Partial-Cross-Correlation Analysis: A New Method for Analyzing Correlations in Complex System. Sci Rep 5, 8143 (2015). https://doi.org/10.1038/srep08143
    """
    R = np.zeros((F.shape[0], F.shape[0]), dtype=float)
    for n in range(F.shape[0]):
        for m in range(n + 1):
            R[n][m] = F[n][m] / np.sqrt(F[n][n] * F[m][m])
            R[m][n] = R[n][m]
    return R


def _cross_correlation(R: np.ndarray):
    """
    Implementation equation (9) from [1]
    [1] Yuan, N., Fu, Z., Zhang, H. et al. Detrended Partial-Cross-Correlation Analysis: A New Method for Analyzing Correlations in Complex System. Sci Rep 5, 8143 (2015). https://doi.org/10.1038/srep08143
    """
    P = np.zeros((R.shape[0], R.shape[0]), dtype=float)
    Cinv = np.linalg.inv(R)
    for n in range(R.shape[0]):
        for m in range(n + 1):
            if Cinv[n][n] * Cinv[m][m] < 0:
                print(f" Error: Sqrt(-1)! No P array values for this S!")
                break
            P[n][m] = -Cinv[n][m] / np.sqrt(Cinv[n][n] * Cinv[m][m])
            P[m][n] = P[n][m]
        else:
            continue
        break
    return P


def _detrend(current_signal: np.ndarray, pd: np.int32):
    """Returns detrended data for dpcca ananlysis
    Args:
        current_signal (np.ndarray): Array with original data or data with time lags.
        pd (np.int32): polynomial degree.

    Returns:
        y_detrended(np.ndarray): Detrended data array.
    """
    current_signal = np.asarray(current_signal, dtype=np.float64)
    n = len(current_signal)
    if n < pd + 2:
        return None
    xw = np.arange(n, dtype=np.int32)
    try:
        p_fit = np.polyfit(xw, current_signal, deg=pd, rcond=None)
        z_fit = np.polyval(p_fit, xw)
        return current_signal - z_fit
    except np.linalg.LinAlgError:
        return None


# @profile()
def dpcca_worker(
    s: Union[int, Iterable],
    arr: Union[np.ndarray, None],
    step: float,
    pd: int,
    gc_params: tuple = None,
    short_vectors=False,
    n_integral=1,
) -> Union[tuple, None]:
    """
    Core of DPCCA algorithm. Takes bunch of S-values and returns 3 3d-matrices,
    where [first index, second index, third index], where [S value, value of signal 1, value of signal 2].

    Args:
    s (Union[int, Iterable]): points where  fluctuation function F(s) is calculated.
    arr (ndarray): dataset array.
    step (float): share of S - value.
    pd (np.int32): polynomial degree.
    gc_params (tuple, optional): _description_. Defaults to None.
    short_vectors (bool, optional): _description_. Defaults to False.
    n_integral (int, optional): Number of cumsum operation before computation. Defaults to 1.
    """
    gc.set_threshold(10, 2, 2)
    s_current = [s] if not isinstance(s, Iterable) else s

    cumsum_arr = arr
    for _ in range(n_integral):
        cumsum_arr = np.cumsum(cumsum_arr, axis=1)

    shape = arr.shape

    F = np.zeros((len(s_current), shape[0], shape[0]), dtype=float)
    R = np.zeros((len(s_current), shape[0], shape[0]), dtype=float)
    P = np.zeros((len(s_current), shape[0], shape[0]), dtype=float)

    for s_i, s_val in enumerate(s_current):

        window_start_indices = np.arange(
            0, shape[1] - s_val + 1, int(step * s_val)
        )  # array of starting indeces of sliding windows
        Xw = np.arange(s_val, dtype=int)
        Y = np.zeros((shape[0], len(window_start_indices)), dtype=object)
        signal_view = np.lib.stride_tricks.sliding_window_view(
            cumsum_arr, s_val, axis=1
        )
        signal_view = signal_view[:, :: int(step * s_val)]
        for n in range(cumsum_arr.shape[0]):
            for m_i, W in enumerate(signal_view[n]):
                if len(W) == 0:
                    print(f"\tFor s = {s_val} W is an empty slice!")
                    return P, R, F
                p = np.polyfit(Xw, W, deg=pd)
                Z = np.polyval(p, Xw)
                Y[n][m_i] = Z - W
                if gc_params is not None:
                    if n % gc_params[0] == 0:
                        gc.collect(gc_params[1])

        Y = np.array([np.concatenate(Y[i]) for i in range(Y.shape[0])])

        F[s_i] = _covariation(Y)
        R[s_i] = _correlation(F[s_i])
        P[s_i] = _cross_correlation(R[s_i])

    return P, R, F


def tds_dpcca_worker(
    s: Union[int, Iterable],
    arr: np.ndarray,
    step: float,
    pd: int,
    time_delays: Union[int, Iterable] = None,
    max_time_delay: int = None,
    gc_params: tuple = None,
    n_integral: int = 1,
) -> Union[tuple, None]:
    """
    Core of DPCAA algorithm with time lags. Takes bunch of S-values and returns 3 4d-matrices: first index
    represents length of time lags array. There is global data and indices: data in all input array as s_val.
    Comparison of signal_1 and signal_2 with time delays by windows(s_val) shifting: find correlation and etc
    of x[i] and y[i+tau] and x[i] and y[i-tau] where tau is value of time lag.

    Args:
        s (Union[int, Iterable]): points where  fluctuation function F(s) is calculated.
        arr (ndarray): dataset array.
        step (float): share of S - value.
        pd (np.int32): polynomial degree.
        time_delays (Union [int, Iterable]): array with time lags.
        max_time_delay (int): value of max time lag.
        gc_params (tuple, optional): _description_. Defaults to None.
        n_integral (int, optional): Number of cumsum operation before computation. Defaults to 1.

    Raises:
        ValueError: Time window couldnt be larger then input data array.
        ValueError: Use lags.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: [P, R, F], where
        [P,R,F] is 4d-matrices, where [lag value, S value, signal 1, signal 2], where
        p is a partial cross-correlation levels on different time scales, coefficients can be used
            to characterize the `intrinsic` relations between the two time series, where one time series is ahead of the other
            on time scales of S.
        r is a coefficients matrix represents the level of cross-correlation on time scales of S.
            However, it should be noted that it only shows the relations between two time series, where one time series
            is ahead of the other.
            This may provide spurious correlation information if the two time series are both correlated with other signals.
        f is a covariance matrix (covariance between any two residuals on each scale).

    """

    if max_time_delay is None:
        return dpcca_worker(s, arr, step, pd, gc_params, n_integral=n_integral)

    s_list = [s] if isinstance(s, int) else list(s)

    if time_delays is not None:
        time_delay_list = np.arange(-time_delays, time_delays + 1, dtype=int)
    elif max_time_delay is not None:
        time_delay_list = np.arange(-max_time_delay, max_time_delay + 1, dtype=int)
    else:
        raise ValueError("Use lags")

    n_lags = len(time_delay_list)
    n_signals, n = arr.shape

    cumsum_arr = arr
    for _ in range(n_integral):
        cumsum_arr = np.cumsum(cumsum_arr, axis=1)

    f = np.zeros((n_lags, len(s_list), n_signals, n_signals), dtype=np.float64)
    r = np.zeros((n_lags, len(s_list), n_signals, n_signals), dtype=np.float64)
    p = np.zeros((n_lags, len(s_list), n_signals, n_signals), dtype=np.float64)

    for s_i, s_val in enumerate(s_list):

        if s_val > n:
            raise ValueError("Time window couldnt be larger then input data array")

        start = np.arange(0, n - s_val + 1)  # all indecies of start time windows
        start_window = start[:: int(step * s_val)]  # with step
        n_windows = len(start_window)  # value of windows

        signal_view = np.lib.stride_tricks.sliding_window_view(
            cumsum_arr, window_shape=s_val, axis=1
        )
        signal_view = signal_view[
            :, :: int(step * s_val), :
        ]  # sliding window in time windows

        for lag_index, lag in enumerate(time_delay_list):
            covariation = np.zeros((n_signals, n_signals), dtype=float)
            correlation = np.zeros((n_signals, n_signals), dtype=float)
            cross_correlation = np.zeros((n_signals, n_signals), dtype=float)

            for w in range(n_windows):
                start_pos = start_window[w]  # start of current window
                start_pos_lag = start_pos + lag
                if start_pos_lag < 0 or start_pos_lag + s_val > n:
                    continue

                signal_windows = np.zeros((n_signals, s_val), dtype=float)
                signal_lag_windows = np.zeros((n_signals, s_val), dtype=float)

                detrend = True
                for sig_idx in range(n_signals):
                    data_true = signal_view[sig_idx, w, :]
                    data_lag_true = cumsum_arr[
                        sig_idx, start_pos_lag : start_pos_lag + s_val
                    ]  # data with lag
                    detrended_true = _detrend(data_true, pd)
                    detrended_lag = _detrend(data_lag_true, pd)

                    if detrended_true is None or detrended_lag is None:
                        detrend = False
                        break

                    signal_windows[sig_idx] = (
                        detrended_true  # detrended signal without lag
                    )
                    signal_lag_windows[sig_idx] = (
                        detrended_lag  # detrended signal with lag
                    )

                if detrend is False:
                    continue  # if hasn't enougth data in current window for detrend->skip this window

                covariation = _covariation(signal_windows, signal_lag_windows)
                correlation = _correlation(covariation)
                cross_correlation = _cross_correlation(correlation)
            f[lag_index, s_i] = covariation
            r[lag_index, s_i] = correlation
            p[lag_index, s_i] = cross_correlation

    return p, r, f


def concatenate_3d_matrices(p: np.ndarray, r: np.ndarray, f: np.ndarray):
    P = np.concatenate(p, axis=1)[0]
    R = np.concatenate(r, axis=1)[0]
    F = np.concatenate(f, axis=1)[0]
    return P, R, F


def dpcca(
    arr: np.ndarray,
    pd: int,
    step: float,
    s: Union[int, Iterable],
    max_lag=None,
    buffer=None,
    gc_params: tuple = None,
    short_vectors: bool = False,
    n_integral: int = 1,
    processes: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Implementation of the Detrended Partial-Cross-Correlation Analysis method proposed by Yuan, N. et al.[1]

    Basic usage:
        You can get whole F(s) function for first vector as:
        ```python
            s_vals = [i**2 for i in range(1, 5)]
            P, R, F, S = dpcaa(input_array, 2, 0.5, s_vals, len(s_vals))
            fluct_func = [F[s][0][0] for s in s_vals]
        ```
    [1] Yuan, N., Fu, Z., Zhang, H. et al. Detrended Partial-Cross-Correlation Analysis:
        A New Method for Analyzing Correlations in Complex System. Sci Rep 5, 8143 (2015).
        https://doi.org/10.1038/srep08143

    Args:
        arr (ndarray): dataset array
        pd (int): polynomial degree
        step (float): share of S - value. It's set usually as 0.5. The integer part of the number will be taken
        s (Union[int, Iterable]): points where  fluctuation function F(s) is calculated. More on that in the article.
        max_lag (int, optional): value of max time lag. Defaults to None.
        processes (int, optional): num of workers to spawn. Defaults to 1.
        buffer (Union[bool, SharedBuffer], optional): Deprecated. Do not considered. Defaults to False.
        gc_params (tuple, optional): _description_. Defaults to None.
        short_vectors (bool, optional): _description_. Defaults to False.
        n_integral (int, optional): Number of cumsum operation before computation. Defaults to 1.

    Raises:
        ValueError: All input S values are larger than vector shape / 4.
        ValueError: Cannot use S > L / 4.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: [P, R, F^2, S], where
            P is a partial cross-correlation levels on different time scales, coefficients can be used
                to characterize the `intrinsic` relations between the two time series on time scales of S.
            R is a coefficients matrix represents the level of cross-correlation on time scales of S.
                However, it should be noted that it only shows the relations between two time series.
                This may provide spurious correlation information if the two time series are both correlated with other signals.
            F^2 is a covariance matrix (covariance between any two residuals on each scale),
            S is used scales.
    """
    if buffer is not None:
        warnings.warn(
            message="Parameter buffer is deprecated",
            category=DeprecationWarning,
            stacklevel=2,
        )

    if max_lag is not None:
        concatenate_all = False  # concatenate if 1d array , no need to use 3d P, R, F
        if arr.ndim == 1:
            arr = np.array([arr])
            concatenate_all = True

        if isinstance(s, Iterable):
            init_s_len = len(s)

            s = list(filter(lambda x: x <= arr.shape[1] / 4, s))
            if len(s) < 1:
                raise ValueError(
                    "All input S values are larger than vector shape / 4 !"
                )

            if len(s) != init_s_len:
                print(f"\tDPCAA warning: only following S values are in use: {s}")

        elif isinstance(s, (float, int)):
            if s > arr.shape[1] / 4:
                raise ValueError("Cannot use S > L / 4")
            s = (s,)

        if max_lag is not None:
            p, r, f = tds_dpcca_worker(
                s,
                arr,
                step,
                pd,
                time_delays=None,
                max_time_delay=max_lag,
                gc_params=gc_params,
                n_integral=n_integral,
            )

            if concatenate_all:
                return concatenate_3d_matrices(p, r, f) + (s,)
            else:
                return p, r, f, s

    if short_vectors:
        return dpcca_worker(
            s,
            arr,
            step,
            pd,
            gc_params=gc_params,
            short_vectors=True,
            n_integral=n_integral,
        ) + (s,)

    concatenate_all = False  # concatenate if 1d array , no need to use 3d P, R, F
    if arr.ndim == 1:
        arr = np.array([arr])
        concatenate_all = True

    if isinstance(s, Iterable):
        init_s_len = len(s)

        s = list(filter(lambda x: x <= arr.shape[1] / 4, s))
        if len(s) < 1:
            raise ValueError("All input S values are larger than vector shape / 4 !")

        if len(s) != init_s_len:
            print(f"\tDPCAA warning: only following S values are in use: {s}")

    elif isinstance(s, (float, int)):
        if s > arr.shape[1] / 4:
            raise ValueError("Cannot use S > L / 4")
        s = (s,)

    if processes == 1 or len(s) == 1:
        p, r, f = dpcca_worker(
            s, arr, step, pd, gc_params=gc_params, n_integral=n_integral
        )
        if concatenate_all:
            return concatenate_3d_matrices(p, r, f) + (s,)

        return p, r, f, s

    processes = len(s) if processes > len(s) else processes

    S = np.array(s, dtype=int) if not isinstance(s, np.ndarray) else s
    S_by_workers = np.array_split(S, processes)

    with closing(Pool(processes=processes)) as pool:
        pool_result = pool.map(
            partial(
                dpcca_worker,
                arr=arr,
                step=step,
                pd=pd,
                gc_params=gc_params,
                n_integral=n_integral,
            ),
            S_by_workers,
        )

    P, R, F = np.array([]), np.array([]), np.array([])

    for res in pool_result:
        P = res[0] if P.size < 1 else np.vstack((P, res[0]))
        R = res[1] if R.size < 1 else np.vstack((R, res[1]))
        F = res[2] if F.size < 1 else np.vstack((F, res[2]))

    if concatenate_all:
        return concatenate_3d_matrices(P, R, F) + (s,)

    return P, R, F, s
