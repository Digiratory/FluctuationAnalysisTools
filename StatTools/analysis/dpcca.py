import gc
from collections.abc import Iterable
from contextlib import closing
from ctypes import c_double
from functools import partial
from multiprocessing import Pool
from typing import Union

import numpy as np

from StatTools.auxiliary import SharedBuffer


def _covariation(signal: np.ndarray):
    """
    Implementation equation (4) from [1]

    [1] Yuan, N., Fu, Z., Zhang, H. et al. Detrended Partial-Cross-Correlation Analysis: A New Method for Analyzing Correlations in Complex System. Sci Rep 5, 8143 (2015). https://doi.org/10.1038/srep08143
    """
    F = np.zeros((signal.shape[0], signal.shape[0]), dtype=float)
    for n in range(signal.shape[0]):
        for m in range(n + 1):
            F[n][m] = np.mean(signal[n] * signal[m])
            signal[m][n] = signal[n][m]
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
    flag_use_lags: bool = False,  # флаг есть ли временные задержки
    gc_params: tuple = None,
    n_integral: int = 1,
) -> tuple[tuple, None]:

    if not flag_use_lags:
        return dpcca_worker(s, arr, step, pd, gc_params, n_integral=n_integral)

    s_current = [s] if isinstance(s, int) else list(s)

    if time_delays is not None:
        time_delay_list = np.array(time_delays, dtype=int)
    elif max_time_delay is not None:
        time_delay_list = np.arange(-max_time_delay - 1, max_time_delay, dtype=int)
    else:
        raise ValueError("должны быть лаги")

    n_lags = len(time_delay_list)
    N = arr.shape[0]

    cumsum_arr = arr
    for _ in range(n_integral):
        cumsum_arr = np.cumsum(cumsum_arr, axis=1)  # интегральная сумма
    F = np.zeros((n_lags, len(s_current), N, N), dtype=np.float64)  # ковариация
    R = np.zeros(
        (n_lags, len(s_current), N, N), dtype=np.float64
    )  # уровни кросс корреляции
    P = np.zeros(
        (n_lags, len(s_current), N, N), dtype=np.float64
    )  # частичная кросс корреляция

    for s_i, s_val in enumerate(s_current):
        signal_view = np.lib.stride_tricks.sliding_window_view(
            cumsum_arr, window_shape=s_val, axis=1
        )
        signal_view = signal_view[:, :: int(step * s_val)]
        n_windows = signal_view.shape[
            1
        ]  # идти по стобцам каждого окна(общее число скользящих окон)
        Xw = np.arange(s_val, dtype=int)  # ранжирование временных масштабов
        Y_detrended = np.zeros_like(signal_view, dtype=np.float64)
        for n in range(cumsum_arr.shape[0]):  # массив по каждому сигналу
            for w_i in range(n_windows):  # массив по каждому окну временному
                W = signal_view[
                    n, w_i
                ]  # значение конкертного сигнала в конкретном временном окне
                p = np.polyfit(Xw, W, deg=pd)  # нахождение фита для тренда
                Z = np.polyval(p, Xw)  # фит для тренда
                Y_detrended[n, w_i] = W - Z
        for lag_i, tau in enumerate(time_delay_list):
            if tau == 0:
                Y1 = Y_detrended
                Y2 = Y_detrended
                current_windows = n_windows
            elif tau > 0:
                Y1 = Y_detrended[
                    :, :-tau
                ]  # сдвиг первого сигнала (остается прежним сигналом)
                Y2 = Y_detrended[:, tau:]  # отделяется часть с начала равная сдвигу
                current_windows = (
                    n_windows - tau
                )  # сдвиг для окна(выбор оставшихся окон)
            else:
                tau_minus = -tau
                if tau_minus >= n_windows:
                    continue  # проверка чтобы лаг был не больше колличества окон
                Y1 = Y_detrended[:, tau_minus:]
                Y2 = Y_detrended[:, :-tau_minus]
                current_windows = n_windows - tau_minus
            if current_windows <= 0:
                continue  # проверка чтобы лаг был не больше колличества окон
            Y1_flat = Y1.reshape(N, -1)  # представление в одномерном массиве
            Y2_flat = Y2.reshape(N, -1)
            # F = np.zeros((Y1_flat, Y2_flat), dtype=float)
            # for i in Y1_flat:
            #     for j in Y2_flat:
            #         F[i][j] = np.mean(i*j)
            #         covariation=F[i][j]
            covariation = np.zeros((N, N), dtype=np.float64)
            for i in range(N):
                for j in range(N):
                    covariation[i, j] = np.mean(Y1_flat[i] * Y2_flat[j])
            F[lag_i, s_i] = covariation
            R[lag_i, s_i] = _correlation(covariation)
            P[lag_i, s_i] = _cross_correlation(R[lag_i, s_i])
    return P, R, F


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
