import gc
from collections.abc import Iterable
from contextlib import closing
from ctypes import c_double
from functools import partial
from multiprocessing import Pool
from typing import Union

import numpy as np

from StatTools.auxiliary import SharedBuffer


def _covariation(signal_1: np.ndarray, signal_2: np.ndarray = None):
    """
    Implementation equation (4) from [1]

    [1] Yuan, N., Fu, Z., Zhang, H. et al. Detrended Partial-Cross-Correlation Analysis: A New Method for Analyzing Correlations in Complex System. Sci Rep 5, 8143 (2015). https://doi.org/10.1038/srep08143
    """
    if signal_2 is None:
        signal_2 = signal_1
    F = np.zeros((signal_1.shape[0], signal_1.shape[0]), dtype=float)
    for n in range(signal_1.shape[0]):
        for m in range(signal_2.shape[0]):
            F[n][m] = np.mean(signal_1[n] * signal_2[m])
            F[m][n] = F[n][m]
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
    Core of DPCAA algorithm. Takes bunch of S-values and returns 3 3d-matrices: first index
    represents S value.
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

        V = np.arange(0, shape[1] - s_val + 1, int(step * s_val))
        Xw = np.arange(s_val, dtype=int)
        Y = np.zeros((shape[0], len(V)), dtype=object)
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


def tdc_dpcca_worker(
    s: Union[int, Iterable],
    arr: np.ndarray,
    step: float,
    pd: int,
    time_delays: Union[int, Iterable] = None,
    max_time_delay: int = None,
    flag_use_lags: bool = False,  # flag if there are time delays
    gc_params: tuple = None,
    n_integral: int = 1,
) -> Union[tuple, None]:
    """
    Core of DPCAA algorithm with time lags. Takes bunch of S-values and returns 3 4d-matrices: first index
    represents length of time lags array.
    """

    if not flag_use_lags:
        return dpcca_worker(s, arr, step, pd, gc_params, n_integral=n_integral)

    s_list = [s] if isinstance(s, int) else list(s)

    if time_delays is not None:
        time_delay_list = np.array(time_delays, dtype=int)
    elif max_time_delay is not None:
        time_delay_list = np.arange(0, max_time_delay + 1, dtype=int)
    else:
        raise ValueError("use lags")

    n_lags = len(time_delay_list)  # length of input time lags array
    n_signals, n = arr.shape

    cumsum_arr = arr
    for _ in range(n_integral):
        cumsum_arr = np.cumsum(cumsum_arr, axis=1)  # integral sum

    f = np.zeros(
        (n_lags, len(s_list), n_signals, n_signals), dtype=np.float64
    )  # covariation
    r = np.zeros(
        (n_lags, len(s_list), n_signals, n_signals), dtype=np.float64
    )  # levels of cross correlation
    p = np.zeros(
        (n_lags, len(s_list), n_signals, n_signals), dtype=np.float64
    )  # partial cross correlation levels

    for s_i, s_val in enumerate(s_list):

        if s_val > n:
            raise ValueError("time window couldnt be larger then input data array")

        start = np.arange(
            0, n - s_val + 1
        )  # all indices of beginning of the windows in input data array
        start_window = start[
            :: int(step * s_val)
        ]  # biginning of the all windows with step
        n_windows = len(start_window)  # value of windows

        signal_view = np.lib.stride_tricks.sliding_window_view(
            cumsum_arr, window_shape=s_val, axis=1
        )  # sliding window
        signal_view = signal_view[
            :, :: int(step * s_val), :
        ]  # (signals,all windows, len of window)

        xw = np.arange(s_val, dtype=int)  # ranging time scales
        y_detrended = np.zeros_like(signal_view, dtype=np.float64)
        for signal in range(cumsum_arr.shape[0]):
            for w_i in range(n_windows):  # array by all indices of all windows
                W = signal_view[
                    signal, w_i
                ]  # value of current signal in current window
                p_fit = np.polyfit(xw, W, deg=pd)
                z_fit = np.polyval(p_fit, xw)
                y_detrended[signal, w_i] = W - z_fit  # detrend
        # We have global data and indices: data in all input array
        # local data: data and indices in current window
        # compare signal_1 and signal_2 with time delays bu indices: find correlation and etc of
        # x[i] and y[i+tau] where tau is value of time lag. Also we have limits:
        # index of current value (i)<=(lenght of input data array(n))-(tau):global_index_lag
        # local length in window of array of analyxed points with lags: length of current window(s_val)-tau:local_max_len_lag
        #
        for lag_index, lag in enumerate(time_delay_list):
            data_points = [[] for _ in range(n_signals)]
            data_lag_points = [[] for _ in range(n_signals)]
            for w in range(n_windows):
                start_pos = start_window[w]  # value of start position of current window
                global_index_lag = n - lag
                if (
                    start_pos > global_index_lag
                ):  # start position cant be larger then global index lag
                    continue
                local_max_len_lag = s_val - lag
                if local_max_len_lag <= 0:
                    continue
                true_points = (
                    global_index_lag - start_pos
                )  # global array of points to analyse
                cross_points = min(
                    true_points, local_max_len_lag
                )  # find the value of pairs of points to analyse considering
                # local and global constraints to avoid dimensional errors
                if cross_points <= 0:
                    continue
                for sig in range(n_signals):
                    data_true = y_detrended[
                        sig, w, :cross_points
                    ]  # detrended array with selected data
                    data_lag_true = y_detrended[
                        sig, w, lag : cross_points + lag
                    ]  # detrended array with selected data with time lags
                    data_points[sig].extend(data_true)
                    data_lag_points[sig].extend(data_lag_true)

            signal_1 = np.array(data_points, dtype=np.float64)
            signal_2 = np.array(data_lag_points, dtype=np.float64)

            covariation = _covariation(signal_1, signal_2)
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
    processes: int = 1,
    buffer: Union[bool, SharedBuffer] = False,
    gc_params: tuple = None,
    short_vectors: bool = False,
    n_integral: int = 1,
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
            s,
            arr,
            step,
            pd,
            gc_params=gc_params,
            n_integral=n_integral,
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
                pd=pd,
                step=step,
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
