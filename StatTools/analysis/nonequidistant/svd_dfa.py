from __future__ import annotations

from collections.abc import Iterable
from typing import Optional, Tuple, Union

import numpy as np


# Helpers: profile, SVD filter
def _as_2d(arr: np.ndarray) -> np.ndarray:
    """Приводит вход к виду (n_signals, N)."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 2:
        return arr
    raise ValueError("arr must be 1D or 2D array")


def _integral_profile(x: np.ndarray) -> np.ndarray:
    """
    Интегральный профиль для DFA:
        y(t) = sum_{i<=t} (x(i) - mean(x))
    Векторизовано по сигналам.
    """
    x = x - np.mean(x, axis=1, keepdims=True)
    return np.cumsum(x, axis=1)


def _trajectory_matrix(y: np.ndarray, L: int) -> np.ndarray:
    """
    Траекторная (Hankel) матрица из профиля y:
      Y[i, j] = y[i + j], i=0..L-1, j=0..K-1
    Где K = N - L + 1

    Возвращает массив формы (n_signals, L, K).
    """
    n_signals, N = y.shape
    if not (2 <= L <= N):
        raise ValueError(f"L must be in [2, N], got L={L}, N={N}")
    K = N - L + 1
    # sliding_window_view: (n_signals, K, L) -> transpose to (n_signals, L, K)
    Y = np.lib.stride_tricks.sliding_window_view(y, window_shape=L, axis=1)
    # Y shape: (n_signals, K, L)
    return np.transpose(Y, (0, 2, 1))


def _diagonal_averaging(Y: np.ndarray) -> np.ndarray:
    """
    Диагональное усреднение (Hankelization) для восстановления ряда из траекторной матрицы.
    Вход: Y формы (n_signals, L, K)
    Выход: y_hat формы (n_signals, N), где N = L + K - 1
    """
    n_signals, L, K = Y.shape
    N = L + K - 1
    y_hat = np.zeros((n_signals, N), dtype=float)
    counts = np.zeros(N, dtype=float)

    # Диагонали с одинаковой суммой индексов s = i + j
    for i in range(L):
        # вклад строки i идёт в позиции s = i .. i+K-1
        y_hat[:, i : i + K] += Y[:, i, :]
        counts[i : i + K] += 1.0

    y_hat /= counts[None, :]
    return y_hat


def _svd_filter_profile(y: np.ndarray, L: int, p: int) -> np.ndarray:
    """
    SVD-фильтрация интегрального профиля y:
    1) строим траекторную матрицу
    2) SVD
    3) зануляем первые p сингулярных компонент
    4) восстанавливаем профиль диагональным усреднением
    """
    if p < 0:
        raise ValueError("p must be >= 0")
    Y = _trajectory_matrix(y, L=L)  # (n_signals, L, K)
    n_signals, L0, K = Y.shape

    # Считаем SVD отдельно для каждого сигнала
    y_filt = np.empty_like(y)
    for idx in range(n_signals):
        Ui, si, Vti = np.linalg.svd(
            Y[idx], full_matrices=False
        )  # Ui:(L,r), si:(r,), Vti:(r,K)
        r = si.size
        p_eff = min(p, r)

        # обнуляем первые p_eff сингулярных значений
        si_f = si.copy()
        si_f[:p_eff] = 0.0

        Yi_f = (Ui * si_f[None, :]) @ Vti  # (L,K)
        y_filt[idx] = _diagonal_averaging(Yi_f[None, :, :])[0]  # back to (N,)

    return y_filt


def _dfa_fluctuation_worker(y: np.ndarray, n: int, m: int) -> np.ndarray:
    """
    DFA флуктуационная функция для одного масштаба n:
      1) делим профиль на сегменты длины n (используем только целое число сегментов)
      2) в каждом сегменте аппроксимируем тренд полиномом порядка m
      3) считаем RMS отклонений, затем усредняем по сегментам
    Вход y: (n_signals, N)
    Выход: F(n) для каждого сигнала: (n_signals,)
    """
    n_signals, N = y.shape
    if n < 2:
        raise ValueError("n must be >= 2")
    nseg = N // n
    if nseg < 2:
        raise ValueError(
            f"Too large scale n={n} for length N={N} (need at least 2 segments)"
        )

    # режем хвост, чтобы укладывалось в целые сегменты
    y0 = y[:, : nseg * n]  # (n_signals, nseg*n)
    seg = y0.reshape(n_signals, nseg, n)  # (n_signals, nseg, n)

    t = np.arange(n, dtype=float)  # (n,)
    # Полиномиальная аппроксимация по каждому сегменту:
    # делаем цикл по сигналам (обычно их мало), внутри — по сегментам векторно.
    F = np.empty(n_signals, dtype=float)

    for i in range(n_signals):
        # polyfit по сегментам: быстрее и проще сделать в цикле по сегментам,
        # но сегментов бывает много — используем lstsq с общей матрицей.
        # Матрица дизайна для полинома порядка m: [1, t, t^2, ...]
        A = np.vander(t, N=m + 1, increasing=True)  # (n, m+1)

        # Решаем (A @ c ≈ seg) для каждого сегмента отдельно.
        # seg_i: (nseg, n) -> транспонируем, чтобы решать пачкой через lstsq по одному RHS
        # Здесь проще и надёжнее цикл по сегментам (nseg до ~N/n), обычно норм.
        rms2 = np.empty(nseg, dtype=float)
        for k in range(nseg):
            coeffs, *_ = np.linalg.lstsq(A, seg[i, k], rcond=None)
            fit = A @ coeffs
            resid = seg[i, k] - fit
            rms2[k] = np.mean(resid * resid)

        F[i] = np.sqrt(np.mean(rms2))

    return F


def svd_dfa(
    arr: np.ndarray,
    s: Union[int, Iterable[int]],
    *,
    integrate: bool = True,
    L: Optional[int] = None,
    p: int = 2,
    m: int = 1,
    n_min: int = 10,
    n_max: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SVD-DFA: возвращает флуктуационную функцию F(n) (без оценки Херста внутри).

    Алгоритм:
      1) (опционально) строим интегральный профиль y(t)
      2) SVD-фильтрация профиля через траекторную матрицу:
         - строим embedding с окном L
         - делаем SVD
         - зануляем p крупнейших сингулярных компонент
         - восстанавливаем профиль диагональным усреднением
      3) DFA: по каждому масштабу n считаем F(n) как RMS отклонений от локального тренда (полином порядка m)

    Args:
        arr:
            1D (N,) или 2D (n_signals, N)
        s:
            масштабы (n) — int или список int.
            Если список не задан, можно передать, например, range(10, N//4).
        integrate:
            True -> сначала делаем DFA-профиль (cumsum(x-mean)).
            False -> работаем с входом как с профилем y(t).
        L:
            размер окна embedding для SVD. Если None -> L = N//3 (как в отчёте).
        p:
            число зануляемых сингулярных компонент (обычно 1..3).
        m:
            порядок полинома для DFA детрендинга в окнах (обычно 1 или 2).
        n_min:
            минимальный масштаб, если ты передаёшь слишком маленькие n.
        n_max:
            максимальный масштаб. Если None -> N//4.

    Returns:
        (F_values, scales):
          - scales: (n_scales,)
          - F_values:
              * если один сигнал: (n_scales,)
              * если несколько:    (n_signals, n_scales)
    """
    x = _as_2d(arr)
    n_signals, N = x.shape

    if N < 50:
        raise ValueError(
            "Series is too short for DFA/SVD-DFA (need at least ~50 points)"
        )

    # масштабы
    if isinstance(s, int):
        scales = np.array([s], dtype=int)
    else:
        scales = np.array(list(s), dtype=int)

    if scales.size == 0:
        raise ValueError("Empty scales list")

    if n_max is None:
        n_max = N // 4
    scales = scales[(scales >= max(2, n_min)) & (scales <= n_max)]
    scales = np.unique(scales)

    if scales.size == 0:
        raise ValueError(
            f"All scales were filtered out. Try wider range, N={N}, n_max={n_max}"
        )

    # профиль
    y = _integral_profile(x) if integrate else x.copy()

    # параметры SVD
    if L is None:
        L = max(2, N // 3)
    if not (2 <= L <= N):
        raise ValueError(f"Invalid L={L} for N={N}")

    # SVD-фильтрация
    y_f = _svd_filter_profile(y, L=L, p=p)

    # DFA: F(n)
    F = np.empty((n_signals, scales.size), dtype=float)
    for j, n in enumerate(scales):
        F[:, j] = _dfa_fluctuation_worker(y_f, n=int(n), m=int(m))

    if arr.ndim == 1:
        return F[0], scales
    return F, scales